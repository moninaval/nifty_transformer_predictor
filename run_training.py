import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os

from config.settings import Config
from src.data_loader import NiftyDataset
from src.model import NiftyPredictor
from src.utils import save_checkpoint, load_checkpoint, ScalerManager

def train_model():
    """
    Main function to set up and run the training process for the Nifty Predictor.
    """
    # --- 1. Configuration and Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Loading and Preprocessing ---
    scaler_manager = ScalerManager(Config.SCALER_DIR)
    
    # Check if dummy data exists, if not, prompt user to run generator
    if not os.path.exists(Config.DATA_FILE):
        print(f"Data file not found at {Config.DATA_FILE}.")
        print("Please run 'python generate_dummy_data.py' first to create dummy data.")
        return

    full_dataset = NiftyDataset(
        data_path=Config.DATA_FILE,
        input_features_dim=Config.INPUT_FEATURES_DIM,
        output_features_dim=Config.OUTPUT_FEATURES_DIM,
        encoder_sequence_length=Config.ENCODER_SEQUENCE_LENGTH,
        decoder_sequence_length=Config.DECODER_SEQUENCE_LENGTH,
        scaler_manager=scaler_manager,
        train_mode=True # Fit scalers in training mode
    )

    # Split dataset into training and validation sets chronologically
    # For time series, a simple random_split is not ideal, but for initial practice, it's okay.
    # For production, implement a strict chronological split as discussed.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False) # No shuffle for time series
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False) # No shuffle for time series

    # --- 3. Model Initialization ---
    model = NiftyPredictor().to(device)
    
    # --- 4. Loss Function and Optimizer ---
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # --- 5. Checkpointing and Early Stopping Setup ---
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Load checkpoint if exists
    start_epoch, best_val_loss = load_checkpoint(Config.BEST_MODEL_CHECKPOINT, model, optimizer)

    # --- 6. Training Loop ---
    print("Starting training...")
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        model.train() # Set model to training mode
        train_loss = 0
        for batch_idx, (src_data, tgt_data) in enumerate(train_loader):
            src_data, tgt_data = src_data.to(device), tgt_data.to(device)

            optimizer.zero_grad() # Clear gradients

            # Forward pass
            # The model predicts a sequence of `decoder_sequence_length` candles
            # Each candle has `output_features_dim` features (OHLCV)
            predictions = model(src_data, tgt_data) # tgt_data is used here for its shape, not its values directly for prediction

            # Reshape target_data to match predictions for loss calculation
            # predictions shape: (batch_size, dec_seq_len, output_features_dim)
            # tgt_data shape: (batch_size, dec_seq_len, output_features_dim)
            loss = criterion(predictions, tgt_data)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- 7. Validation ---
        model.eval() # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad(): # Disable gradient calculations for validation
            for src_data_val, tgt_data_val in val_loader:
                src_data_val, tgt_data_val = src_data_val.to(device), tgt_data_val.to(device)
                
                predictions_val = model(src_data_val, tgt_data_val)
                loss_val = criterion(predictions_val, tgt_data_val)
                val_loss += loss_val.item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- 8. Checkpointing and Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch + 1, best_val_loss, Config.BEST_MODEL_CHECKPOINT)
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {Config.PATIENCE} epochs.")
                break

    print("Training finished.")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")

if __name__ == "__main__":
    # Ensure the necessary directories are created
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SCALER_DIR, exist_ok=True)
    train_model()
