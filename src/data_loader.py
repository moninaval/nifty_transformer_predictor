import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import ta # Technical Analysis library

from config.settings import Config
from src.utils import ScalerManager

class NiftyDataset(Dataset):
    """
    Custom Dataset for Nifty 5-minute candle data.
    Handles loading, feature engineering, scaling, and sequence creation.
    """
    def __init__(self, data_path, input_features_dim, output_features_dim,
                 encoder_sequence_length, decoder_sequence_length,
                 scaler_manager, train_mode=True):
        """
        Args:
            data_path (str): Path to the raw CSV data file.
            input_features_dim (int): Number of features for the Transformer Encoder input.
            output_features_dim (int): Number of features for the Decoder target output (OHLCV).
            encoder_sequence_length (int): Length of the input sequence for the encoder.
            decoder_sequence_length (int): Length of the output sequence to predict by the decoder.
            scaler_manager (ScalerManager): Instance of ScalerManager for saving/loading scalers.
            train_mode (bool): True if in training mode (fit scalers), False for inference (load scalers).
        """
        self.data_path = data_path
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        self.enc_seq_len = encoder_sequence_length
        self.dec_seq_len = decoder_sequence_length
        self.scaler_manager = scaler_manager
        self.train_mode = train_mode

        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """
        Loads the raw CSV data, calculates features, handles missing values,
        and prepares the data for scaling and sequence creation.
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path, parse_dates=['Timestamp'], index_col='Timestamp')
        df = df.sort_index()

        # --- 1. Handle Missing Values (Simple Forward-fill for demonstration) ---
        # In a real scenario, you'd have more sophisticated handling for gaps.
        # For lower-frequency features (macro, FII/DII), ensure they are forward-filled
        # from their original frequency to 5-min before saving to CSV, or implement here.
        df = df.ffill().bfill() # Forward fill then backward fill any remaining NaNs

        # --- 2. Feature Engineering (Simplified for demonstration) ---
        # This is where you'd implement the calculation of all 100 features.
        # For this example, I'll assume many are already in the dummy data,
        # and add a few common ones using the `ta` library.
        print("Calculating technical indicators and derived features...")

        # NIFTY Technical Indicators (example subset)
        df['MA_5'] = ta.trend.sma_indicator(df['NIFTY_Close'], window=5)
        df['MA_20'] = ta.trend.sma_indicator(df['NIFTY_Close'], window=20)
        df['RSI_14'] = ta.momentum.rsi(df['NIFTY_Close'], window=14)
        macd = ta.trend.MACD(df['NIFTY_Close'])
        df['MACD_Line'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['ATR'] = ta.volatility.average_true_range(df['NIFTY_High'], df['NIFTY_Low'], df['NIFTY_Close'])
        
        # Time-based features (cyclical encoding)
        df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['Minute_sin'] = np.sin(2 * np.pi * df.index.minute / 60)
        df['Minute_cos'] = np.cos(2 * np.pi * df.index.minute / 60)
        
        # Day of week (one-hot encoding) - assuming Monday=0, Sunday=6
        df['Day_Mon'] = (df.index.dayofweek == 0).astype(int)
        df['Day_Tue'] = (df.index.dayofweek == 1).astype(int)
        df['Day_Wed'] = (df.index.dayofweek == 2).astype(int)
        df['Day_Thu'] = (df.index.dayofweek == 3).astype(int)
        df['Day_Fri'] = (df.index.dayofweek == 4).astype(int)

        # Ensure all 100 features are present.
        # This part assumes your dummy/actual CSV has columns matching the 100 features.
        # If not, you'll need to add more feature engineering here.
        # For now, we'll select the first `input_features_dim` columns after initial processing.
        # You MUST ensure your CSV columns align with your 100 features.
        
        # Drop any rows that might have NaN after indicator calculation (e.g., initial rows for MAs)
        df = df.dropna()

        # --- 3. Define Input and Target Features ---
        # The order of features here must match the order in your collected data
        # for consistent indexing.
        
        # Target features (OHLCV of Nifty)
        self.target_cols = ['NIFTY_Open', 'NIFTY_High', 'NIFTY_Low', 'NIFTY_Close', 'NIFTY_Volume']
        
        # Input features (all 100 features)
        # This is a placeholder. You need to define the exact 100 column names
        # in the order they appear in your processed DataFrame.
        # For this example, we'll just take the first `input_features_dim` columns after dropping NaNs.
        # In a real scenario, explicitly list all 100 feature names.
        
        # Example: If your df has more than 100 columns after feature engineering,
        # you'd select the specific 100 you want.
        # For this example, we'll just take the first `input_features_dim` columns
        # assuming the dummy data generator creates enough.
        
        # Make sure the target columns are not part of the input features if they are to be predicted directly.
        # However, for Transformer, the input features are historical, and target is future.
        # So, the current NIFTY_OHLCV are input features, but the future NIFTY_OHLCV are targets.
        
        # Get all column names from the DataFrame after dropping NaNs
        all_feature_cols = df.columns.tolist()
        
        # Ensure that the number of available features is at least INPUT_FEATURES_DIM
        if len(all_feature_cols) < self.input_features_dim:
            raise ValueError(f"Not enough features in the processed data. Expected {self.input_features_dim}, but found {len(all_feature_cols)}. Please ensure your dummy data or actual data generation includes all 100 features.")
            
        # Select the input features. For simplicity, taking the first `input_features_dim` columns.
        # In a real project, you would explicitly define `self.input_cols = [...]` with your 100 feature names.
        self.input_cols = all_feature_cols[:self.input_features_dim]
        
        self.data = df[self.input_cols].values # All input features
        self.targets = df[self.target_cols].values # Only OHLCV for targets

        # --- 4. Scaling ---
        self._scale_data()

        # --- 5. Sequence Creation ---
        self._create_sequences()

    def _scale_data(self):
        """
        Scales input features and target features using MinMaxScaler.
        Scalers are fitted in train_mode and loaded in inference mode.
        """
        print("Scaling data...")
        input_scaler_file = Config.INPUT_SCALER_FILE
        target_scaler_file = Config.TARGET_SCALER_FILE

        if self.train_mode:
            self.input_scaler = MinMaxScaler(feature_range=(0, 1))
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))

            # Fit scalers on the entire dataset for training
            self.scaled_data = self.input_scaler.fit_transform(self.data)
            self.scaled_targets = self.target_scaler.fit_transform(self.targets)

            # Save scalers
            self.scaler_manager.save_scaler(self.input_scaler, os.path.basename(input_scaler_file))
            self.scaler_manager.save_scaler(self.target_scaler, os.path.basename(target_scaler_file))
        else:
            # Load pre-fitted scalers for inference
            self.input_scaler = self.scaler_manager.load_scaler(os.path.basename(input_scaler_file))
            self.target_scaler = self.scaler_manager.load_scaler(os.path.basename(target_scaler_file))

            self.scaled_data = self.input_scaler.transform(self.data)
            self.scaled_targets = self.target_scaler.transform(self.targets)

    def _create_sequences(self):
        """
        Creates input-target sequences for the Transformer.
        Each input sequence is `enc_seq_len` long.
        Each target sequence is `dec_seq_len` long, starting from the next time step.
        """
        self.input_sequences = []
        self.target_sequences = []

        # The loop runs up to the point where we can extract a full encoder sequence
        # and a full decoder target sequence.
        # The target sequence for a given input sequence starts from the (encoder_sequence_length)th element
        # and extends for decoder_sequence_length.
        
        # Total length needed for one (input, target) pair:
        # enc_seq_len (for encoder input) + dec_seq_len (for decoder target)
        # The target for X[i:i+enc_seq_len] is Y[i+enc_seq_len : i+enc_seq_len+dec_seq_len]

        total_len = len(self.scaled_data)
        for i in range(total_len - self.enc_seq_len - self.dec_seq_len + 1):
            encoder_input = self.scaled_data[i : i + self.enc_seq_len]
            # The target for the decoder is the OHLCV of the *future* candles
            decoder_target = self.scaled_targets[i + self.enc_seq_len : i + self.enc_seq_len + self.dec_seq_len]

            self.input_sequences.append(torch.tensor(encoder_input, dtype=torch.float32))
            self.target_sequences.append(torch.tensor(decoder_target, dtype=torch.float32))

        print(f"Created {len(self.input_sequences)} sequences.")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        """
        Returns a single (input_sequence, target_sequence) pair.
        """
        return self.input_sequences[idx], self.target_sequences[idx]
