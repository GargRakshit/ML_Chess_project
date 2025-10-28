# fen_dataset.py
import torch
from torch.utils.data import Dataset, get_worker_info
from chess import Board
import math
import os
import time

try:
    # --- Use the original 13-channel function ---
    from auxiliary_func import board_to_matrix
except ImportError:
    print("Error: auxiliary_func.py missing board_to_matrix."); raise

class FENDataset(Dataset):
    """ Lazy loads from preprocessed TSV file using offsets. """
    def __init__(self, tsv_file_path, max_eval_clip=1500):
        self.tsv_file_path = tsv_file_path
        self.max_eval_clip = max_eval_clip
        self.line_offsets = []
        self.num_lines = 0
        self.valid_items = 0
        self.invalid_items = 0

        if not os.path.exists(tsv_file_path): raise FileNotFoundError(f"TSV file not found: {tsv_file_path}")

        print(f"Indexing TSV: {tsv_file_path}...")
        start_time = time.time()
        try:
            with open(self.tsv_file_path, 'rb') as f:
                _ = f.readline() # Skip header
                offset = f.tell()
                line = f.readline()
                while line:
                    self.line_offsets.append(offset)
                    self.num_lines += 1
                    offset = f.tell()
                    line = f.readline()
            end_time = time.time()
            print(f"Indexed {self.num_lines:,} positions in {end_time - start_time:.2f}s.")
        except Exception as e: print(f"Error indexing file: {e}"); raise
        if self.num_lines == 0: print(f"Warning: No data lines found in {tsv_file_path}.")

    def __len__(self): return self.num_lines

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_lines:
            if self.num_lines == 0: raise IndexError("Empty dataset.")
            idx = 0 # Fallback

        line = None; offset = self.line_offsets[idx]
        try:
            with open(self.tsv_file_path, 'r', encoding='utf-8') as f:
                f.seek(offset)
                line = f.readline().strip()
            if not line: raise ValueError(f"Read empty line at idx {idx}")
            parts = line.split('\t')
            if len(parts) != 2: raise ValueError(f"Invalid TSV line at idx {idx}")
            fen_string, evaluation_str = parts
            evaluation = int(evaluation_str)
        except Exception as e:
            return None # Indicate failure

        try:
            board = Board(fen_string)
            # --- Use the 13-channel function ---
            board_tensor = torch.tensor(board_to_matrix(board), dtype=torch.float32)
        except Exception: # Catch broader errors (invalid FEN, legal_moves fail)
            return None # Indicate invalid FEN or board state

        clipped_eval = max(min(evaluation, self.max_eval_clip), -self.max_eval_clip)
        normalized_eval = torch.tanh(torch.tensor(clipped_eval / 100.0 / 10.0, dtype=torch.float32))
        target_tensor = normalized_eval.unsqueeze(0)
        return board_tensor, target_tensor