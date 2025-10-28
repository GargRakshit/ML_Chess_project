# finetune_eval_script.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, get_worker_info
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import os
import gc
import multiprocessing # Import multiprocessing

# Import your custom modules
try:
    from evaluation_model import ChessEvaluationModel, load_pretrained_weights
    from fen_dataset import FENDataset # Lazy TSV version
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure evaluation_model.py and fen_dataset.py are accessible.")
    raise SystemExit

# --- Configuration ---
PRETRAINED_MODEL_PATH = "models/TORCH_100EPOCHS.pth"
FILTERED_TSV_PATH = "fen_data/first_1.5GB_evaluations.tsv"
SAVE_MODEL_PATH = "models/finetuned_model_1.pth"

NUM_CLASSES_ORIGINAL = 1884 # Correct value

# --- Training ---
BATCH_SIZE = 512
LEARNING_RATE = 1e-5
EPOCHS = 5
VAL_SPLIT = 0.10
MAX_EVAL_CLIP = 1500

# --- DataLoader ---
NUM_WORKERS = 7  # Reduced number of workers
PIN_MEMORY = True
PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
PREFETCH_FACTOR = 2  # Reduced prefetch factor
# ---------------------

# --- Custom Collate Function ---
def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    try:
        result = torch.utils.data.dataloader.default_collate(batch)
        return result
    except Exception:
        return None, None
# ----------------------------

def main_training():
    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f'Using device: {device}'); gc.collect()

    # --- Initialize Model (13-channel) ---
    model = ChessEvaluationModel().to(device)

    # --- Load Weights (Loads conv1, conv2, fc1) ---
    try: model = load_pretrained_weights(model, PRETRAINED_MODEL_PATH, NUM_CLASSES_ORIGINAL)
    except Exception as e: print(f"Weight loading failed: {e}. Continuing.")

    # --- Compile Model ---
    print("\nCompiling model..."); start_compile = time.time()
    try:
        if hasattr(torch, 'compile'):
            compiled_model = torch.compile(model, mode='reduce-overhead')
            _ = compiled_model(torch.randn(2, 13, 8, 8).to(device)) # Test with 13 channels
            model = compiled_model; print(f"Compiled successfully in {time.time() - start_compile:.2f}s.")
        else: print("torch.compile not available.")
    except Exception as e: print(f"Compile failed: {e}. Continuing.")

    # --- Prepare Dataset ---
    print("\nIndexing dataset..."); start_index = time.time()
    try:
        full_dataset = FENDataset(tsv_file_path=FILTERED_TSV_PATH, max_eval_clip=MAX_EVAL_CLIP)
        print(f"Indexing took {time.time() - start_index:.2f}s.")
    except Exception as e: print(f"Dataset init failed: {e}"); raise SystemExit
    if len(full_dataset) == 0: print("Dataset empty!"); raise SystemExit

    # --- Split ---
    print(f"Total dataset size: {len(full_dataset):,}")
    
    # Validate first item
    first_item = full_dataset[0]
    if first_item is None or not isinstance(first_item[0], torch.Tensor):
        print("Error: Failed to load first dataset item")
        raise SystemExit
    
    val_size = int(len(full_dataset) * VAL_SPLIT)
    val_size = max(1, val_size) if len(full_dataset) > 1 else 0
    train_size = len(full_dataset) - val_size
    if train_size <= 0: print("No training samples!"); raise SystemExit
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)
    print(f"\nTrain size: {len(train_ds):,}, Val size: {len(val_ds):,}")
    del full_dataset; gc.collect()

    # --- DataLoaders ---
    use_prefetch = PREFETCH_FACTOR if NUM_WORKERS > 0 else None
    use_persistent = PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    use_pin = PIN_MEMORY if NUM_WORKERS > 0 and device.type == 'cuda' else False
    print("\nInitializing data loaders...")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_pin, 
                         persistent_workers=use_persistent, drop_last=True, prefetch_factor=use_prefetch, collate_fn=collate_fn_skip_none)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin,
                       persistent_workers=use_persistent, drop_last=False, prefetch_factor=use_prefetch, collate_fn=collate_fn_skip_none)

    # --- Loss, Optimizer, Scaler ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(init_scale=2.**16, enabled=(device.type == 'cuda'))
    print(f"\nAMP enabled: {scaler.is_enabled()}")

    # --- Training Loop ---
    print(f"\nStarting fine-tuning for {EPOCHS} epochs...")
    print("-------------------------")
    
    # Use standard tqdm (not notebook) for scripts
    epoch_pbar = tqdm(range(EPOCHS), desc="Overall Epoch Progress", unit="epoch")

    for epoch in epoch_pbar:
        epoch_start = time.time(); model.train(); train_loss = 0.0; batches_ok = 0
        train_b_pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False, unit="batch")

        for i, batch in enumerate(train_b_pbar):
            if batch[0] is None:
                continue
            try: 
                inputs, targets = batch
            except Exception: 
                continue
            if not isinstance(inputs, torch.Tensor):
                continue

            inputs, targets = inputs.to(device, non_blocking=use_pin), targets.to(device, non_blocking=use_pin)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                try: outputs = model(inputs); loss = criterion(outputs, targets)
                except Exception: continue
                if not torch.isfinite(loss): continue

            try: scaler.scale(loss).backward()
            except RuntimeError: optimizer.zero_grad(set_to_none=True); continue

            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            train_loss += loss.item(); batches_ok += 1
            if batches_ok > 0: train_b_pbar.set_postfix(loss=f'{train_loss / batches_ok:.6f}')

        avg_train_loss = train_loss / batches_ok if batches_ok > 0 else float('nan')

        # --- Validation ---
        model.eval(); val_loss = 0.0; val_batches_ok = 0
        val_b_pbar = tqdm(val_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for i_val, batch_val in enumerate(val_b_pbar):
                if batch_val[0] is None: continue
                try: inputs, targets = batch_val
                except Exception as e: print(f"Val unpack error: {e}"); continue
                if not isinstance(inputs, torch.Tensor): continue

                inputs, targets = inputs.to(device, non_blocking=use_pin), targets.to(device, non_blocking=use_pin)

                with autocast(enabled=scaler.is_enabled()):
                    try: outputs = model(inputs); loss = criterion(outputs, targets)
                    except Exception as e: print(f"Val Fwd/Loss error: {e}"); continue

                if torch.isfinite(loss):
                     val_loss += loss.item(); val_batches_ok += 1
                     if val_batches_ok > 0: val_b_pbar.set_postfix(loss=f'{val_loss / val_batches_ok:.6f}')

        avg_val_loss = val_loss / val_batches_ok if val_batches_ok > 0 else float('nan')
        epoch_time_min = (time.time() - epoch_start) / 60
        epoch_pbar.set_postfix(TrainLoss=f"{avg_train_loss:.5f}", ValLoss=f"{avg_val_loss:.5f}")
        print(f"\nEpoch {epoch + 1}/{EPOCHS} | Time: {epoch_time_min:.2f} min | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        del inputs, targets, outputs, loss; gc.collect(); torch.cuda.empty_cache()

    print("\nFine-tuning finished.")
    print("-------------------------")
    
    # --- Save Model ---
    try:
        save_dir = os.path.dirname(SAVE_MODEL_PATH)
        if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir); print(f"Created dir: {save_dir}")
        state_dict = model.state_dict()
        torch.save(state_dict, SAVE_MODEL_PATH)
        print(f"\nModel saved to: {SAVE_MODEL_PATH}")
    except Exception as e: print(f"\nError saving model: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # Necessary for Windows multiprocessing
    main_training()