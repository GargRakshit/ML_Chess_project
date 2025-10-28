# evaluation_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from model import ChessModel
except ImportError:
    print("Error: Could not import ChessModel from model.py.")
    raise

class ChessEvaluationModel(nn.Module):
    """
    Evaluation model with 13 input channels, matching the original model.
    """
    def __init__(self):
        super(ChessEvaluationModel, self).__init__()
        # --- Uses 13 input channels ---
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        # -----------------------------
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.value_head = nn.Linear(256, 1)
        self._init_weights()

    def _init_weights(self):
        # Initialize all layers
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu'); nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu'); nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.value_head.weight); nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.value_head(x)
        x = torch.tanh(x)
        return x

def load_pretrained_weights(evaluation_model, pretrained_path, num_classes_original):
    """
    Loads weights for conv1, conv2, and fc1 from the original model.
    """
    print(f"Loading pre-trained weights from: {pretrained_path}")
    original_model_structure = ChessModel(num_classes_original)
    try:
        device = next(evaluation_model.parameters()).device
        pretrained_state_dict = torch.load(pretrained_path, map_location=device)
        original_model_structure.load_state_dict(pretrained_state_dict)
    except Exception as e:
        print(f"Error loading original state dict: {e}"); raise

    eval_model_dict = evaluation_model.state_dict()
    weights_to_copy = {}; copied=set(); skipped=set()

    for key, param in original_model_structure.state_dict().items():
        prefix = key.split('.')[0]
        if key in eval_model_dict and eval_model_dict[key].shape == param.shape:
            # --- Copy conv1, conv2, and fc1 ---
            if prefix == 'conv1' or prefix == 'conv2' or prefix == 'fc1':
                weights_to_copy[key] = param
                copied.add(prefix)
            else:
                skipped.add(prefix)
        else:
            skipped.add(prefix) # fc2 or other mismatches

    if not weights_to_copy:
        print("\nWarning: No weights were copied! Check layer names/shapes.")
    else:
        eval_model_dict.update(weights_to_copy)
        try:
            evaluation_model.load_state_dict(eval_model_dict)
            print(f"OK loaded: {list(copied)}")
        except RuntimeError as e:
            print(f"RuntimeError loading partial state dict: {e}")

    expected_skips = {'fc2'}; actual_skips = skipped - copied
    if expected_skips.intersection(actual_skips):
        print(f"Skipped (expected): {list(expected_skips.intersection(actual_skips))}")
    if actual_skips - expected_skips:
         print(f"Warning: Unexpected skips: {list(actual_skips - expected_skips)}")

    return evaluation_model