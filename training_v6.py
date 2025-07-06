import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm.auto import tqdm
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import glob
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import faiss
from datetime import timedelta
import time
from visualization import debug_example  # Import the debug function

class QuantumWalkRetriever(nn.Module):
    def __init__(self, embedding_dim, k=8, hidden_dim=128, walk_steps=3):
        super().__init__()
        # Embedder is part of the model for batched GPU encoding
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        for p in self.embedder.parameters(): p.requires_grad = False
        self.k = k
        self.walk_steps = walk_steps
        d = embedding_dim
        
        # Coin network for transition probabilities
        self.coin_net = nn.Sequential(
            nn.Linear(d*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k)
        )
        
        # Path scoring network - V6: Input dimension changed!
        # Input: [initial_sentence_emb (d), final_walk_state (k), question_emb (d)]
        path_net_input_dim = d + k + d # 2*d + k
        self.path_net = nn.Sequential(
            nn.Linear(path_net_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Initialize weights
        # Re-initialize first layer due to changed input size
        nn.init.xavier_uniform_(self.path_net[0].weight)
        nn.init.zeros_(self.path_net[0].bias)
        nn.init.xavier_uniform_(self.path_net[2].weight)
        nn.init.zeros_(self.path_net[2].bias)

    def build_adjacency_matrix(self, n_ex, neighbors):
        """Build a sparse adjacency matrix from neighbor indices."""
        if n_ex == 0:
            return torch.sparse_coo_tensor(torch.zeros((2, 1)), torch.ones(1), (1, 1))
            
        # Convert neighbors to tensor if not already
        if not isinstance(neighbors, torch.Tensor):
            neighbors = torch.tensor(neighbors, dtype=torch.long)
            
        # Create self-loops for all nodes
        row_indices = torch.arange(n_ex, device=neighbors.device)
        col_indices = torch.arange(n_ex, device=neighbors.device)
        
        # Add edges from neighbors
        if neighbors.numel() > 0:
            # Create indices for valid neighbors
            valid_mask = (neighbors >= 0) & (neighbors < n_ex)
            valid_nbrs = neighbors[valid_mask]
            
            if valid_nbrs.numel() > 0:
                # Create row indices for neighbors
                nbr_rows = torch.arange(neighbors.size(0), device=neighbors.device)
                nbr_rows = nbr_rows.unsqueeze(1).expand(-1, neighbors.size(1))
                nbr_rows = nbr_rows[valid_mask]
                
                # Combine self-loops and neighbor edges
                row_indices = torch.cat([row_indices, nbr_rows])
                col_indices = torch.cat([col_indices, valid_nbrs])
        
        # Create sparse tensor
        indices = torch.stack([row_indices, col_indices])
        values = torch.ones(len(row_indices), device=neighbors.device)
        return torch.sparse_coo_tensor(indices, values, (n_ex, n_ex))

    def forward(self, questions: List[str], sent_embs: List[torch.Tensor], 
                neighbors: List[torch.Tensor], labels: List[torch.Tensor]):
        # Handle empty batch case
        if not questions or not sent_embs or not neighbors or not labels:
            return []
            
        # Filter out examples with empty embeddings
        valid_examples = []
        valid_questions = []
        valid_embs = []
        valid_neighbors = []
        valid_labels = []
        
        for i, (q, emb, nbr, lbl) in enumerate(zip(questions, sent_embs, neighbors, labels)):
            if emb.size(0) > 0:  # Only process non-empty examples
                valid_examples.append(i)
                valid_questions.append(q)
                valid_embs.append(emb)
                valid_neighbors.append(nbr)
                valid_labels.append(lbl)
        
        if not valid_examples:
            return []
            
        # Batch encode all valid questions
        try:
            q_embs = self.embedder.encode(valid_questions, convert_to_tensor=True, 
                                        device=next(self.parameters()).device)
        except Exception as e:
            print(f"Error in question encoding: {e}")
            return []
        
        final_logits_batch = []
        
        # NOTE: Processing examples individually in a loop due to variable graph sizes per example.
        # This prevents full batch vectorization of the walk but handles heterogeneous inputs.
        # Consider graph batching libraries (e.g., PyG, DGL) for potential optimization if performance is critical.
        for idx, (qv, emb_ex, nbrs, lbl) in enumerate(zip(q_embs, valid_embs, valid_neighbors, valid_labels)):
            try:
                n_ex = emb_ex.size(0)
                if n_ex == 0:
                    continue
                
                # Build sparse adjacency matrix
                A = self.build_adjacency_matrix(n_ex, nbrs).to(emb_ex.device)
                
                # Initialize state (Uniform initialization)
                state = torch.ones(n_ex, self.k, dtype=torch.float32, device=emb_ex.device) / np.sqrt(n_ex * self.k)
                
                # Compute coin amplitudes based on initial sentence and question
                base_inp = torch.cat([emb_ex, qv.unsqueeze(0).expand_as(emb_ex)], dim=1)
                amps = self.coin_net(base_inp)
                
                # Perform walk steps
                for _ in range(self.walk_steps):
                    # Apply coin operator
                    new_state = state * amps
                    
                    # Apply shift operator
                    new_state = new_state.view(n_ex, -1)
                    with autocast(device_type='cuda', enabled=False):
                        new_state = torch.sparse.mm(A.float(), new_state.float())
                    new_state = new_state.view(n_ex, self.k) # Shape: (n_ex, k)
                    
                    # Normalize
                    norm = new_state.norm()
                    state = new_state / norm if norm > 0 else new_state
                
                # --- V6 Output Calculation ---
                final_state = state # State after the walk loop, shape (n_ex, k)
                
                # Create input for path_net: [initial_emb, final_state, question_emb]
                path_net_input = torch.cat([
                    emb_ex,                        # Shape (n_ex, d)
                    final_state,                  # Shape (n_ex, k)
                    qv.unsqueeze(0).expand(n_ex, -1) # Shape (n_ex, d)
                ], dim=1) # Result shape (n_ex, 2*d + k)
                
                # Calculate final logits using path_net
                final_logits = self.path_net(path_net_input).squeeze(-1) # Shape (n_ex, 1) -> (n_ex,)
                final_logits_batch.append(final_logits)
                # -----------------------------

                # OLD V5 output:
                # logits_batch.append(state.abs().sum(dim=1))
                
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                continue
        
        return final_logits_batch

    def compute_knn_neighbors(self, queries, k):
        """Compute k-NN neighbors for a batch of queries using chunked processing"""
        chunk_size = 64  # Further reduced chunk size
        all_distances = []
        all_indices = []
        
        for i in range(0, len(queries), chunk_size):
            chunk = queries[i:i + chunk_size]
            try:
                distances, indices = self.index.search(chunk, k)
                all_distances.append(distances)
                all_indices.append(indices)
            except Exception as e:
                print(f"Error in chunk {i//chunk_size}: {e}")
                # Fallback to CPU for this chunk
                cpu_index = faiss.IndexFlatL2(self.sent_embs.shape[1])
                cpu_index.add(self.sent_embs)
                distances, indices = cpu_index.search(chunk, k)
                all_distances.append(distances)
                all_indices.append(indices)
        
        return np.concatenate(all_distances), np.concatenate(all_indices)

# (Rest of training, dataset, evaluation unchanged, embedding_dim passed from dataset)")}

class HotpotDataset(Dataset):
    def __init__(self, data_file, embeddings_dir, is_train=True, dataset_percentage=100, k=8):
        print(f"Loading data file: {data_file}")
        with open(data_file, 'r') as f:
            data_full = json.load(f)
        
        # Calculate how many examples to use based on percentage
        total_examples = len(data_full)
        num_examples = int(total_examples * (dataset_percentage / 100))
        
        # Load precomputed embeddings and metadata - do this BEFORE data slicing
        prefix = "train" if is_train else "dev"
        print(f"Loading embeddings from {embeddings_dir}")
        self.sent_embs = np.load(os.path.join(embeddings_dir, f"{prefix}_embeddings.npy"))
        self.doc_offsets_full = np.load(os.path.join(embeddings_dir, f"{prefix}_offsets.npy"))
        
        # Only slice data AFTER loading offsets
        print(f"Using {num_examples} out of {total_examples} examples ({dataset_percentage}%)")
        
        # Ensure consistency: slice both data and offsets to the same length
        data_subset = data_full[:num_examples]
        offsets_subset = self.doc_offsets_full[:num_examples]
        
        # Filter out empty examples
        print("Filtering out empty examples...")
        valid_indices = []
        for idx in range(len(data_subset)):
            start, end = offsets_subset[idx]
            if start < end:  # Only keep non-empty examples
                valid_indices.append(idx)
        
        # Store selected data and corresponding offsets
        self.data = [data_subset[i] for i in valid_indices]
        self.doc_offsets = offsets_subset[valid_indices]
        print(f"Filtered to {len(self.data)} non-empty examples")
        
        # Check if k-NN neighbors have already been precomputed and saved
        neighbors_file = os.path.join(embeddings_dir, f"{prefix}_neighbors_k{k}.npy")
        
        if os.path.exists(neighbors_file):
            print(f"Loading precomputed k-NN neighbors from {neighbors_file}")
            neighbors_full = np.load(neighbors_file)
            self.neighbors = neighbors_full
            print("k-NN neighbors loaded from file")
        else:
            # Precompute k-NN neighbors for each example's sentences using PyTorch
            print(f"Computing k-NN neighbors with k={k} using PyTorch...")
            self.neighbors = np.zeros((len(self.sent_embs), k), dtype=np.int64)
            
            # Convert embeddings to PyTorch tensor
            embeddings_tensor = torch.from_numpy(self.sent_embs).float()
            
            # Process examples in batches for better performance
            batch_size = 1000  # Process 1000 examples at a time
            for batch_start in tqdm(range(0, len(valid_indices), batch_size), desc="Processing batches"):
                batch_end = min(batch_start + batch_size, len(valid_indices))
                batch_indices = valid_indices[batch_start:batch_end]
                
                # Process all examples in this batch
                for idx, orig_idx in enumerate(batch_indices):
                    start, end = offsets_subset[orig_idx]
                    if start >= end:
                        continue
                        
                    # Get embeddings for this example
                    example_embs = embeddings_tensor[start:end]
                    
                    # Compute pairwise distances using PyTorch
                    distances = torch.cdist(example_embs, example_embs)
                    
                    # Get nearest neighbors, handling cases where there are fewer points than k+1
                    n_points = len(example_embs)
                    actual_k = min(k+1, n_points)  # Don't try to get more neighbors than points
                    _, local_neighbors = torch.topk(distances, actual_k, largest=False)
                    local_neighbors = local_neighbors[:, 1:].numpy()  # Remove self
                    
                    # Pad with self if we have fewer neighbors than k
                    if n_points <= k:
                        padded_neighbors = np.full((n_points, k), np.arange(n_points)[:, None])
                        padded_neighbors[:, :actual_k-1] = local_neighbors
                        local_neighbors = padded_neighbors
                    
                    # Convert to local indices within the example
                    for i in range(len(local_neighbors)):
                        for j in range(k):
                            if local_neighbors[i, j] >= len(example_embs):
                                local_neighbors[i, j] = i  # Replace out-of-bounds with self
                    
                    # Convert local indices back to global indices before storing
                    global_neighbors = local_neighbors + start
                    self.neighbors[start:end] = global_neighbors
            
            # Save the computed neighbors for future runs
            print(f"Saving k-NN neighbors to {neighbors_file}")
            np.save(neighbors_file, self.neighbors)
            print("k-NN neighbors saved to file")
        
        print("Preparing examples...")
        self.examples = self._prepare_examples()
        print(f"Dataset initialization complete with {len(self.examples)} examples")

    def _prepare_examples(self):
        exs = []
        total = len(self.data)
        
        for idx in range(total):
            if idx % 1000 == 0:
                print(f"Preparing examples: {idx}/{total} ({idx/total*100:.1f}%)")
            
            start, end = self.doc_offsets[idx]
            q = self.data[idx]['question']
            
            # Get sentence embeddings and neighbors for this example
            sent_embs = self.sent_embs[start:end]
            
            # Skip examples with empty embeddings
            if sent_embs.shape[0] == 0:
                print(f"Skipping example {idx} with empty sent_embs (start={start}, end={end})")
                continue
                
            local_neighbors = self.neighbors[start:end] - start  # Convert to local indices
            
            # Ensure local_neighbors doesn't go negative (when indices point outside example)
            local_neighbors = np.maximum(local_neighbors, 0)
            
            # Get labels and preserve full context information
            lbls = []
            context = []
            for title, slist in self.data[idx]['context']:
                context.append((title, slist))  # Preserve the full context
                for sid, s in enumerate(slist):
                    lbls.append(1 if [title, sid] in self.data[idx].get('supporting_facts', []) else 0)
            
            # Check if labels match the number of sentences
            if len(lbls) != sent_embs.shape[0]:
                print(f"Warning: Example {idx} has mismatch between labels ({len(lbls)}) and embeddings ({sent_embs.shape[0]})")
                # Adjust labels if needed
                if len(lbls) < sent_embs.shape[0]:
                    # Pad labels with zeros
                    lbls.extend([0] * (sent_embs.shape[0] - len(lbls)))
                else:
                    # Truncate labels
                    lbls = lbls[:sent_embs.shape[0]]
            
            exs.append({
                'question': q,
                'sent_embs': sent_embs,
                'neighbors': local_neighbors,
                'labels': lbls,
                'context': context,  # Add full context
                'supporting_facts': self.data[idx].get('supporting_facts', [])  # Add supporting facts
            })
        
        # Final check - report examples with empty embeddings
        empty_count = 0
        for idx, ex in enumerate(exs):
            if ex['sent_embs'].shape[0] == 0:
                empty_count += 1
        
        if empty_count > 0:
            print(f"WARNING: {empty_count} examples have empty embeddings after preparation!")
            
        return exs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def setup_distributed():
    # For single-GPU training, we don't need distributed setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

# Add this function to handle evaluation during training
def evaluate_model(model, dataloader, device, num_batches=None):
    """Evaluates the model on the given dataloader and returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_examples = 0
    empty_batches = 0
    
    # HotpotQA metrics
    total_support_f1 = 0.0
    total_support_precision = 0.0
    total_support_recall = 0.0
    total_joint_f1 = 0.0
    
    if num_batches is None:
        total_samples = len(dataloader)
    else:
        total_samples = min(num_batches, len(dataloader))
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Evaluating', total=total_samples)):
            if num_batches is not None and i >= num_batches:
                break
                
            # Check if batch has any non-empty examples
            has_valid_data = any(len(ex.get('sent_embs', [])) > 0 for ex in batch)
            if not has_valid_data:
                empty_batches += 1
                if empty_batches % 10 == 0:  # Log every 10th empty batch
                    print(f"Warning: Skipping empty batch {i} during evaluation")
                continue
            
            # Prepare batch data
            questions = [ex['question'] for ex in batch]
            sent_embs = [torch.from_numpy(ex['sent_embs']).to(device) for ex in batch]
            neighbors = [torch.from_numpy(ex['neighbors']).to(device).long() for ex in batch]
            labels = [torch.tensor(ex['labels'], device=device).float() for ex in batch] # Use original 0/1 labels
            
            # Forward pass with autocast
            with autocast(device_type='cuda'):
                # Model now returns final_logits directly
                final_logits_list = model(questions, sent_embs, neighbors, labels)
            
            if not final_logits_list:  # Skip empty batches
                empty_batches += 1
                continue
            
            # Calculate metrics for each example
            batch_loss = 0.0
            batch_support_f1 = 0.0
            batch_support_precision = 0.0
            batch_support_recall = 0.0
            valid_examples = 0
            
            for final_logits, lbl in zip(final_logits_list, labels):
                # Make sure lbl contains the original 0/1 float labels
                if lbl.numel() == 0 or final_logits.numel() == 0 or final_logits.size(0) != lbl.size(0):
                    print(f"Warning: Skipping example due to size mismatch or empty tensors (final_logits: {final_logits.shape}, labels: {lbl.shape})")
                    continue # Skip if sizes mismatch
                
                # Calculate loss using BCEWithLogitsLoss with final_logits
                loss = F.binary_cross_entropy_with_logits(final_logits, lbl, reduction='mean')
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_loss += loss.item()
                    
                    # --- Validation F1 Calculation (uses topk(true_k), keep as is for monitoring) ---
                    if lbl.sum() > 0: # Only calculate F1 if positive labels exist
                        # Get predicted and true supporting facts
                        k = int(lbl.sum().item()) # True count
                        # Need probabilities for topk comparison, calculate from final_logits
                        probs = torch.sigmoid(final_logits.float()) # Sigmoid for BCE scores
                        pred_indices = set(probs.topk(min(k, len(probs)))[1].tolist()) # Predict top true_k
                        true_indices = set(lbl.nonzero().flatten().tolist())
                        
                        # Calculate precision and recall for supporting facts
                        tp = len(pred_indices.intersection(true_indices))
                        if len(pred_indices) > 0:
                            precision = tp / len(pred_indices)
                        else:
                            precision = 0.0
                            
                        if len(true_indices) > 0:
                            recall = tp / len(true_indices)
                        else:
                            recall = 0.0 # Or 1.0 if len(pred_indices)==0 too?
                            
                        # Calculate F1 score
                        if precision + recall > 0:
                            f1 = 2 * (precision * recall) / (precision + recall)
                        else:
                            f1 = 0.0
                        
                        batch_support_f1 += f1
                        batch_support_precision += precision
                        batch_support_recall += recall
                        valid_examples += 1 # Count example only if F1 was calculated
                    # --- End Validation F1 Calculation ---
                else:
                    print(f"Warning: NaN or Inf loss encountered in evaluation.")
            
            if valid_examples > 0:
                # Average metrics over valid examples where F1 was calculated
                batch_loss_avg = batch_loss / len(final_logits_list) # Average loss over all processed examples
                batch_support_f1_avg = batch_support_f1 / valid_examples
                batch_support_precision_avg = batch_support_precision / valid_examples
                batch_support_recall_avg = batch_support_recall / valid_examples
                
                total_loss += batch_loss_avg # Accumulate average batch loss
                total_support_f1 += batch_support_f1_avg
                total_support_precision += batch_support_precision_avg
                total_support_recall += batch_support_recall_avg
                total_batches += 1
                # total_examples += valid_examples # This was double counting
            else:
                # Accumulate loss even if no valid examples for F1 calculation in batch
                if len(final_logits_list) > 0:
                     batch_loss_avg = batch_loss / len(final_logits_list)
                     if not (torch.isnan(batch_loss_avg) or torch.isinf(batch_loss_avg)):
                         total_loss += batch_loss_avg
                         total_batches += 1
                     else:
                         empty_batches += 1 # Count as empty if loss is invalid
                else:
                    empty_batches += 1
    
    # Calculate final metrics
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    support_f1 = total_support_f1 / total_batches if total_batches > 0 else 0.0
    support_precision = total_support_precision / total_batches if total_batches > 0 else 0.0
    support_recall = total_support_recall / total_batches if total_batches > 0 else 0.0
    
    print(f"Evaluation: Processed batches: {total_batches}, Examples contributing to F1: {int(total_support_f1 * total_batches / support_f1 if support_f1 > 0 else 0)}, Skipped/Empty batches: {empty_batches}") # Approx total F1 examples
    print(f"Supporting Facts - F1: {support_f1:.4f}, Precision: {support_precision:.4f}, Recall: {support_recall:.4f}")
    return avg_loss, support_f1, support_precision, support_recall

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized inputs.
    Instead of stacking examples, it returns a list of individual examples.
    """
    return batch  # Just return the batch as a list of dictionaries without any stacking

def print_memory_usage():
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("CUDA not available, memory stats not available")

def load_checkpoint(model, optimizer, checkpoint_path, device, is_distributed=False):
    """Load checkpoint with proper handling of model architecture changes."""
    try:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Checkpoint loaded successfully")
        
        # Get current model state dict - handle DDP case
        if is_distributed:
            current_state_dict = model.module.state_dict()
        else:
            current_state_dict = model.state_dict()
        
        # Get checkpoint state dict
        checkpoint_state_dict = checkpoint['model_state_dict']
        
        # Handle missing keys (new architecture)
        for key in current_state_dict.keys():
            if key not in checkpoint_state_dict:
                print(f"Initializing new parameter: {key}")
                if 'path_net' in key:
                    # Initialize path_net weights
                    if 'weight' in key:
                        nn.init.xavier_uniform_(current_state_dict[key])
                    elif 'bias' in key:
                        nn.init.zeros_(current_state_dict[key])
        
        # Handle size mismatches
        for key in checkpoint_state_dict.keys():
            if key in current_state_dict:
                if checkpoint_state_dict[key].shape != current_state_dict[key].shape:
                    print(f"Resizing parameter: {key}")
                    if 'coin_net.2' in key:  # Handle k parameter change
                        if 'weight' in key:
                            # Resize weight matrix
                            new_weight = torch.zeros_like(current_state_dict[key])
                            min_k = min(checkpoint_state_dict[key].shape[0], current_state_dict[key].shape[0])
                            new_weight[:min_k] = checkpoint_state_dict[key][:min_k]
                            checkpoint_state_dict[key] = new_weight
                        elif 'bias' in key:
                            # Resize bias vector
                            new_bias = torch.zeros_like(current_state_dict[key])
                            min_k = min(checkpoint_state_dict[key].shape[0], current_state_dict[key].shape[0])
                            new_bias[:min_k] = checkpoint_state_dict[key][:min_k]
                            checkpoint_state_dict[key] = new_bias
        
        # Load state dict - handle DDP case
        if is_distributed:
            model.module.load_state_dict(checkpoint_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint_state_dict, strict=False)
        print("Model state loaded")
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        
        # Load epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"Successfully loaded checkpoint from epoch {start_epoch-1}")
        return start_epoch
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--embeddings_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--walk_steps', type=int, default=2)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--eval_batches', type=int, default=50)
    parser.add_argument('--start_from_scratch', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dataset_percentage', type=float, default=100, help='Percentage of dataset to use (1-100)')
    parser.add_argument('--debug_examples', type=int, default=0, help='Number of examples to debug per epoch (0 to disable)')
    parser.add_argument('--debug_output_dir', type=str, default='debug_output', help='Directory to save debug visualizations')
    args = parser.parse_args()

    # Create debug output directory
    os.makedirs(args.debug_output_dir, exist_ok=True)

    # Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # Setup distributed training if enabled
    print("Starting distributed setup...")
    if args.distributed:
        # Get local rank from environment
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        print(f"Initializing process group: local_rank={local_rank}, world_size={world_size}")
        try:
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=local_rank
            )
            print("Process group initialized")
            
            # Set device for this process
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            
            # Adjust batch sizes for distributed training
            args.batch_size = args.batch_size // world_size
            args.eval_batch_size = args.eval_batch_size // world_size
            print(f"Adjusted batch sizes for distributed training: train={args.batch_size}, eval={args.eval_batch_size}")
        except Exception as e:
            print(f"Error in distributed setup: {e}")
            raise
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
    
    print(f"Using device: {device} (local_rank={local_rank})")

    # Create datasets with precomputed embeddings
    print("Loading datasets...")
    try:
        train_dataset = HotpotDataset(args.train_file, args.embeddings_dir, is_train=True, dataset_percentage=args.dataset_percentage, k=args.k)
        print(f"Train dataset loaded with {len(train_dataset)} examples")
        dev_dataset = HotpotDataset(args.dev_file, args.embeddings_dir, is_train=False, dataset_percentage=args.dataset_percentage, k=args.k)
        print(f"Dev dataset loaded with {len(dev_dataset)} examples")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

    # Adjust number of workers based on available resources
    args.num_workers = min(4, os.cpu_count() // world_size if args.distributed else os.cpu_count())
    print(f"Using {args.num_workers} workers for data loading")

    # Create data loaders
    print("Creating data loaders...")
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    print("Data loaders created successfully")

    # Initialize model
    print("Initializing model...")
    print_memory_usage()
    embedding_dim = 384 # Hardcoded for 'all-MiniLM-L6-v2'
    model = QuantumWalkRetriever(
        embedding_dim=embedding_dim, 
        k=args.k,
        hidden_dim=args.hidden_dim,
        walk_steps=args.walk_steps
    ).to(device)
    print_memory_usage()
    
    if args.distributed:
        print("Wrapping model with DDP...")
        # find_unused_parameters=True is necessary because the SentenceTransformer embedder
        # parameters are frozen (requires_grad=False) and thus won't have gradients flowing
        # during the backward pass, which DDP would otherwise error on.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        print("Model wrapped with DDP")

    # Initialize optimizer and scheduler
    print("Initializing optimizer and scheduler...")
    if args.distributed:
        # For DDP, access model attributes through module
        optimizer = optim.AdamW([
            {'params': model.module.coin_net.parameters(), 'lr': args.lr},
            {'params': model.module.path_net.parameters(), 'lr': args.lr}
        ], weight_decay=0.01)
    else:
        optimizer = optim.AdamW([
            {'params': model.coin_net.parameters(), 'lr': args.lr},
            {'params': model.path_net.parameters(), 'lr': args.lr}
        ], weight_decay=0.01)
    
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs-args.warmup_epochs)
    scaler = GradScaler(enabled=True)
    print("Optimizer and scheduler initialized")

    # Load checkpoint if exists
    start_epoch = 0
    if not args.start_from_scratch and os.path.exists(args.checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_epoch_*.pt'))
        if checkpoint_files:
            try:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                print(f"Loading checkpoint: {latest_checkpoint}")
                start_epoch = load_checkpoint(model, optimizer, latest_checkpoint, device, args.distributed)
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0

    print("Starting training loop...")
    # Training loop
    best_eval_loss = float('inf')
    best_support_f1 = 0.0
    best_support_precision = 0.0
    best_support_recall = 0.0
    for epoch in range(start_epoch, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        empty_batches = 0
        
        # Debug counter for this epoch
        debug_count = 0
        
        # === FIX: Iterate over train_dataloader instead of manual chunking ===
        for batch in train_dataloader: # Iterate directly over dataloader
        # =====================================================================
            
            # Check if batch has any non-empty examples (handled by custom_collate_fn?)
            # Assuming custom_collate_fn returns a list of dicts
            has_valid_data = any(len(ex.get('sent_embs', [])) > 0 for ex in batch)
            if not has_valid_data:
                empty_batches += 1
                continue # Skip completely empty batches if any occur

            # Prepare batch data
            questions = [ex['question'] for ex in batch]
            sent_embs = [torch.from_numpy(ex['sent_embs']).to(device) for ex in batch]
            neighbors = [torch.from_numpy(ex['neighbors']).to(device).long() for ex in batch]
            labels = [torch.tensor(ex['labels'], device=device).float() for ex in batch] # Original 0/1 labels
            
            # Filter out examples within the batch that might still be empty individually
            valid_indices = [i for i, emb in enumerate(sent_embs) if emb.size(0) > 0]
            if not valid_indices:
                empty_batches += 1
                continue

            questions = [questions[i] for i in valid_indices]
            sent_embs = [sent_embs[i] for i in valid_indices]
            neighbors = [neighbors[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]

            # Forward pass with autocast
            with autocast(device_type='cuda'):
                # Model forward pass now returns final_logits_list
                final_logits_list = model(questions, sent_embs, neighbors, labels)
            
            if not final_logits_list: # If model returns empty list after internal filtering
                empty_batches += 1
                continue
            
            # Calculate loss for each example returned by the model
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True) # Ensure requires_grad=True
            valid_examples_in_batch = 0
            actual_labels = [labels[i] for i, l in enumerate(final_logits_list)] # Align labels with returned logits

            # Ensure final_logits_list and actual_labels have the same length
            if len(final_logits_list) != len(actual_labels):
                 print(f"Warning: Mismatch between final_logits_list ({len(final_logits_list)}) and actual_labels ({len(actual_labels)}) lengths. Skipping batch.")
                 empty_batches += 1
                 continue

            for final_logits, lbl in zip(final_logits_list, actual_labels):
                 # Ensure lbl is the original 0/1 float tensor
                if lbl.numel() == 0 or final_logits.numel() == 0 or final_logits.size(0) != lbl.size(0):
                    print(f"Warning: Skipping loss calculation due to size mismatch or empty tensors (final_logits: {final_logits.shape}, labels: {lbl.shape})")
                    continue # Skip if sizes mismatch
                
                # Calculate loss using BCEWithLogitsLoss with final_logits
                loss = F.binary_cross_entropy_with_logits(final_logits, lbl, reduction='mean')
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    # Accumulate loss correctly
                    if batch_loss.requires_grad:
                        batch_loss = batch_loss + loss
                    else: # First valid loss in batch
                        batch_loss = loss
                    valid_examples_in_batch += 1
                else:
                    print(f"Warning: NaN or Inf loss encountered for an example. Skipping loss contribution.")

            
            if valid_examples_in_batch > 0:
                # Average loss over valid examples in the batch
                batch_loss = batch_loss / valid_examples_in_batch
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                
                total_train_loss += batch_loss.item()
                train_batches += 1
            else:
                empty_batches += 1
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else float('inf')
        print(f"Epoch {epoch} - Training Loss: {avg_train_loss:.6f}, Processed batches: {train_batches}, Skipped/Empty batches: {empty_batches}")
        
        # LR scheduling
        if epoch >= args.warmup_epochs: # Apply scheduler after warmup
             lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr'] # Get current LR for logging
        print(f"Epoch {epoch} - Current LR: {current_lr:.6e}")

        # Evaluation
        dev_loss, dev_support_f1, dev_support_precision, dev_support_recall = evaluate_model(
            model, 
            dev_dataloader, 
            device=device,
            num_batches=args.eval_batches
        )
        
        print(f"Epoch {epoch} - Validation Loss: {dev_loss:.6f}")
        print(f"Supporting Facts - F1: {dev_support_f1:.4f}, Precision: {dev_support_precision:.4f}, Recall: {dev_support_recall:.4f}")
        
        # Save best model based on F1 score
        if dev_support_f1 > best_support_f1 or (dev_support_f1 == best_support_f1 and dev_loss < best_eval_loss):
            best_eval_loss = dev_loss
            best_support_f1 = dev_support_f1
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            
            # Save the model state dict properly (handle DDP case)
            model_state_dict = model.module.state_dict() if args.distributed else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': dev_loss,
                'support_f1': dev_support_f1,
                'support_precision': dev_support_precision,
                'support_recall': dev_support_recall,
            }, best_model_path)
            print(f"New best model saved to {best_model_path}")
        
        # Save checkpoint
        save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Save the model state dict properly (handle DDP case)
        model_state_dict = model.module.state_dict() if args.distributed else model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': dev_loss,
            'support_f1': dev_support_f1,
            'support_precision': dev_support_precision,
            'support_recall': dev_support_recall,
        }, save_path)
        print(f"Checkpoint saved to {save_path}")

if __name__ == '__main__':
    main() 