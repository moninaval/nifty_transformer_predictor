import os

class Config:
    # Model Hyperparameters
    D_MODEL = 512  # Dimensionality of the model's internal representations
    NUM_HEADS = 8  # Number of attention heads in Multi-Head Attention
    NUM_ENCODER_LAYERS = 6 # Number of Transformer Encoder layers
    NUM_DECODER_LAYERS = 6 # Number of Transformer Decoder layers
    DIM_FEEDFORWARD = 2048 # Dimension of the feed-forward network (typically 4 * D_MODEL)
    DROPOUT = 0.1 # Dropout rate for regularization

    # Data Parameters
    ENCODER_SEQUENCE_LENGTH = 20 # Number of historical 5-min candles for input
    DECODER_SEQUENCE_LENGTH = 6  # Number of future 5-min candles to predict
    INPUT_FEATURES_DIM = 100 # Total number of input features per 5-min candle
    OUTPUT_FEATURES_DIM = 5  # OHLCV features to predict per future candle

    # Training Parameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    NUM_EPOCHS = 100 # Adjust based on dataset size and convergence
    PATIENCE = 10 # For early stopping: number of epochs to wait for improvement

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    SCALER_DIR = os.path.join(BASE_DIR, 'scalers') # Directory to save/load scalers

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    # File names
    DATA_FILE = os.path.join(DATA_DIR, 'nifty_historical_data.csv')
    INPUT_SCALER_FILE = os.path.join(SCALER_DIR, 'input_scaler.pkl')
    TARGET_SCALER_FILE = os.path.join(SCALER_DIR, 'target_scaler.pkl')
    BEST_MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'best_nifty_transformer.pth')
