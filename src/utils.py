import torch
import torch.nn as nn
import math
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in the original Transformer paper.
    This adds information about the position of tokens in the sequence, as attention mechanisms
    are permutation-invariant.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe) # 'pe' is not a model parameter, but part of the model's state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (batch_size, sequence_length, d_model)
        Returns:
            Tensor, shape (batch_size, sequence_length, d_model)
        """
        # Add positional encoding to the input embeddings
        # The positional encoding is broadcasted across the batch dimension
        x = x + self.pe[:, :x.size(1)]
        return x

class ScalerManager:
    """
    Manages saving and loading MinMaxScaler instances.
    Necessary to ensure consistent scaling between training and inference.
    """
    def __init__(self, scaler_path):
        self.scaler_path = scaler_path

    def save_scaler(self, scaler, filename):
        """Saves a fitted scaler to a file."""
        filepath = os.path.join(self.scaler_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {filepath}")

    def load_scaler(self, filename):
        """Loads a scaler from a file."""
        filepath = os.path.join(self.scaler_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded from {filepath}")
        return scaler

def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """Saves the model's state, optimizer's state, and training progress."""
    print(f"Saving checkpoint at epoch {epoch} with validation loss: {val_loss:.4f}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """Loads a checkpoint and restores model and optimizer states."""
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}. Starting from scratch.")
        return 0, float('inf') # Return epoch 0 and infinite loss

    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    print(f"Loaded checkpoint from epoch {epoch} with validation loss: {val_loss:.4f}")
    return epoch, val_loss
