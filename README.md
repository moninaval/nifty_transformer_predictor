# nifty_transformer_predictor
Nifty Transformer Predictor
This project implements a Transformer Encoder-Decoder model for predicting the next 5-minute candles of the Nifty 50 index, leveraging a comprehensive set of 100 features including technical indicators, global market data, macroeconomic factors, and more. The goal is to forecast the Open, High, Low, Close, and Volume (OHLCV) for a sequence of 6 future 5-minute candles.

Project Goal
The primary objective is to build a robust deep learning model that can learn complex temporal patterns and interdependencies from diverse financial datasets to accurately predict short-term movements in the Nifty 50 index.

Features Utilized
The model is designed to consume a rich input vector of 100 features per 5-minute candle, encompassing:

NIFTY Core Price & Volume: OHLCV, Percentage Change, VWAP.

Derived NIFTY Technical Indicators: A wide range of MAs, RSI, MACD, Bollinger Bands, ATR, Stochastic, ROC, CMF, OBV, ADL, Williams %R, CCI, DMI/ADX.

NIFTY Derivatives Data: Futures Open Interest & Volume, Put-Call Ratio, Implied Volatility, Max Pain, and OI Changes for ATM/OTM/ITM options, India VIX.

Global Indices & Volatility: Dow Jones, S&P 500, Nasdaq, Nikkei 225, Hang Seng, FTSE 100, DAX, SGX Nifty, US VIX, Dollar Index (DXY).

Key Commodity Prices: Crude Oil, Gold, Silver, Natural Gas, Global Commodity Index.

Forex / Currency Impact: USD/INR, EUR/INR, GBP/INR Exchange Rates, INR Volatility Index.

Macro-Economic Indicators (India): Repo Rate, Reverse Repo Rate, CPI Inflation, WPI Inflation, Manufacturing PMI, Services PMI.

Institutional Flow: FII Net Buy/Sell, DII Net Buy/Sell, Mutual Fund Equity Inflows, EPFO Investment in Equities.

Sectoral Index Movements (NSE): NIFTY Bank, IT, Pharma, FMCG, Auto, Energy, Realty, Metal, PSU Bank, Financial Services Indices.

Time-Based Features & Event Flags: Cyclical hour/minute, day of week, and binary flags for significant events (Budget, RBI Policy, Geopolitical, US Fed, Election, Corporate Earnings, Monsoon Forecast).

Model Architecture
The core of this project is a Transformer Encoder-Decoder model, adapted for time-series regression:

Input Processing: Each 5-minute candle's 100 features are first passed through a linear projection layer to transform them into a d_model (512-dimensional) embedding.

Positional Encoding: Sinusoidal positional encodings are added to these embeddings to provide temporal order information.

Encoder: Consists of 6 Transformer Encoder layers, processing the historical sequence of ENCODER_SEQUENCE_LENGTH (default 20) candle embeddings.

Decoder: Consists of 6 Transformer Decoder layers. It uses masked self-attention and cross-attention (attending to the Encoder's output) to predict the DECODER_SEQUENCE_LENGTH (default 6) future 5-minute candles.

Output Head: A final linear layer projects the decoder's output from d_model back to the 5 OHLCV features for each predicted candle.

Hyperparameters:

D_MODEL: 512

NUM_HEADS: 8

NUM_ENCODER_LAYERS: 6

NUM_DECODER_LAYERS: 6

DIM_FEEDFORWARD: 2048

DROPOUT: 0.1

ENCODER_SEQUENCE_LENGTH: 20

DECODER_SEQUENCE_LENGTH: 6

INPUT_FEATURES_DIM: 100

OUTPUT_FEATURES_DIM: 5 (OHLCV)

Project Structure
nifty_transformer_predictor/
├── data/
│   └── nifty_historical_data.csv  # Your raw 5-min data (5 years recommended)
├── checkpoints/
│   └── best_nifty_transformer.pth # Saved model checkpoints
├── scalers/
│   ├── input_scaler.pkl         # Scaler for input features
│   └── target_scaler.pkl        # Scaler for target features
├── config/
│   └── settings.py              # Centralized configuration for hyperparameters and paths
├── src/
│   ├── __init__.py              # Python package initializer
│   ├── data_loader.py           # Handles data loading, preprocessing, feature engineering, scaling, and sequence creation
│   ├── model.py                 # Defines the Transformer Encoder-Decoder model architecture
│   ├── utils.py                 # Utility functions (Positional Encoding, Checkpointing, Scaler Management)
├── run_training.py              # Main script to initiate and manage model training
└── generate_dummy_data.py       # Helper script to create synthetic data for testing

Setup and Installation
Clone the repository:

git clone https://github.com/your-username/nifty_transformer_predictor.git
cd nifty_transformer_predictor

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install dependencies:

pip install pandas numpy torch torchvision torchaudio scikit-learn ta

Data
Generate Dummy Data (for quick testing):
To quickly test the setup, you can generate a CSV file with dummy data:

python generate_dummy_data.py

This will create data/nifty_historical_data.csv.

Collect Real Data (for actual training):
For real-world performance, you must replace data/nifty_historical_data.csv with your actual 5-minute Nifty data.

Period: Aim for 5 years of historical 5-minute data.

Features: Ensure your CSV contains all 100 features as discussed, with a consistent Timestamp column. For lower-frequency data (macro, FII/DII), ensure it's forward-filled to 5-minute granularity.

Training
To start the training process:

python run_training.py

The run_training.py script will:

Load and preprocess your data (including feature engineering and scaling).

Initialize the Transformer model.

Use MSELoss as the criterion and Adam as the optimizer.

Implement checkpointing to save the best model weights based on validation loss.

Incorporate early stopping to prevent overfitting.

Print training and validation loss for each epoch.

Inference
After training, the saved model checkpoint (checkpoints/best_nifty_transformer.pth) and scalers (scalers/input_scaler.pkl, scalers/target_scaler.pkl) can be used for making predictions on new, unseen data.

(Details for an inference.py script would be provided in a later stage, but conceptually it involves: loading the model and scalers, acquiring the latest ENCODER_SEQUENCE_LENGTH (20) 5-minute candles, preprocessing them identically to training, feeding them to the model, and inverse-transforming the predictions.)

Future Work and Improvements
Advanced Feature Engineering: Explore more sophisticated ways to incorporate news sentiment, order book data, and inter-market relationships.

Walk-Forward Validation: Implement a rigorous walk-forward backtesting strategy for more realistic performance evaluation.

Hyperparameter Optimization: Use more advanced techniques like Optuna or Weights & Biases for systematic hyperparameter tuning.

Ensemble Methods: Combine predictions from multiple models.

Uncertainty Quantification: Explore methods to quantify the uncertainty of predictions (e.g., quantile regression, Bayesian neural networks).

Deployment Pipeline: Build a robust, automated pipeline for real-time data ingestion, prediction, and monitoring.

Disclaimer
This project is for educational and research purposes only and should not be used for actual financial trading decisions. Financial markets are highly complex and volatile, and past performance is not indicative of future results. Trading involves substantial risk of loss.