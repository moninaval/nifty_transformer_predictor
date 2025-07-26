import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.settings import Config

def generate_dummy_nifty_data(num_days=500, output_path=Config.DATA_FILE):
    """
    Generates dummy 5-minute Nifty-like data with 100 features for testing.
    This simulates 5 years of data (approx 250 trading days/year * 5 years = 1250 days).
    We'll generate for `num_days` for a quicker test.
    """
    print(f"Generating dummy data for {num_days} trading days...")

    start_date = datetime(2020, 1, 1)
    
    # Define base features for Nifty OHLCV
    base_price = 12000
    base_volume = 1000000

    data_rows = []
    current_date = start_date

    # Define 100 feature names
    feature_names = [
        'NIFTY_Open', 'NIFTY_High', 'NIFTY_Low', 'NIFTY_Close', 'NIFTY_Volume',
        'NIFTY_Pct_Change', 'NIFTY_VWAP'
    ] # 7 features

    # Add dummy technical indicator names (25 features)
    for i in range(1, 26):
        feature_names.append(f'TI_{i}')

    # Add dummy derivatives data names (12 features)
    for i in range(1, 13):
        feature_names.append(f'Deriv_{i}')

    # Add dummy global indices & correlation names (10 features)
    for i in range(1, 11):
        feature_names.append(f'Global_{i}')
        
    # Add dummy commodity prices (5 features)
    for i in range(1, 6):
        feature_names.append(f'Commodity_{i}')

    # Add dummy forex/currency impact (4 features)
    for i in range(1, 5):
        feature_names.append(f'Forex_{i}')

    # Add dummy macro-economic indicators (6 features)
    for i in range(1, 7):
        feature_names.append(f'Macro_{i}')

    # Add dummy institutional flow (4 features)
    for i in range(1, 5):
        feature_names.append(f'InstFlow_{i}')

    # Add dummy sectoral index movements (10 features)
    for i in range(1, 11):
        feature_names.append(f'Sector_{i}')

    # Add dummy event flags (5 features, excluding day of week which is calculated in DataLoader)
    feature_names.extend([
        'Is_Budget_Day', 'Is_RBI_Policy_Day', 'Is_Geopolitical_Event',
        'Is_US_Fed_Day', 'Is_Election_Day'
    ])

    # Ensure we have exactly 100 features
    if len(feature_names) != 100:
        raise ValueError(f"Feature names list does not have 100 elements. Found: {len(feature_names)}")

    for day in range(num_days):
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Simulate daily updates for some features
        daily_global_factor = np.random.uniform(0.99, 1.01)
        daily_macro_factor = np.random.uniform(0.995, 1.005)
        daily_inst_flow = np.random.uniform(-1000, 1000) # Example buy/sell

        # Simulate monthly/quarterly updates for some features (very simplified)
        is_first_day_of_month = (current_date.day == 1)
        if is_first_day_of_month:
            monthly_inflation = np.random.uniform(0.04, 0.08)
            monthly_repo_rate = np.random.uniform(0.05, 0.07)
        
        # Simulate event flags (randomly for dummy data)
        is_budget_day = 1 if (day % 100 == 0) else 0 # Every 100 days
        is_rbi_policy_day = 1 if (day % 50 == 0) else 0 # Every 50 days
        is_geopolitical_event = 1 if (day % 200 == 0) else 0
        is_us_fed_day = 1 if (day % 70 == 0) else 0
        is_election_day = 1 if (day % 365 == 0) else 0


        for minute_offset in range(0, 375, 5): # 9:15 AM to 3:30 PM (375 minutes)
            timestamp = current_date + timedelta(hours=9, minutes=15 + minute_offset)

            # Simulate Nifty OHLCV
            open_price = base_price + np.random.uniform(-5, 5)
            close_price = open_price + np.random.uniform(-10, 10)
            high_price = max(open_price, close_price) + np.random.uniform(0, 5)
            low_price = min(open_price, close_price) - np.random.uniform(0, 5)
            volume = base_volume + np.random.uniform(-500000, 500000)
            
            # Ensure volume is non-negative
            volume = max(0, volume)

            # NIFTY_Pct_Change (current 5-min vs previous 5-min) - will be calculated in DataLoader
            # NIFTY_VWAP (current 5-min) - will be calculated in DataLoader

            row_values = [
                open_price, high_price, low_price, close_price, volume,
                np.nan, np.nan # Placeholders for NIFTY_Pct_Change, NIFTY_VWAP
            ]

            # Add dummy values for other 93 features
            # TI_1 to TI_25 (25 features)
            row_values.extend([np.random.uniform(20, 80) for _ in range(25)])
            # Deriv_1 to Deriv_12 (12 features)
            row_values.extend([np.random.uniform(0, 100000) for _ in range(12)])
            # Global_1 to Global_10 (10 features)
            row_values.extend([np.random.uniform(10000, 40000) * daily_global_factor for _ in range(10)])
            # Commodity_1 to Commodity_5 (5 features)
            row_values.extend([np.random.uniform(50, 100) for _ in range(5)])
            # Forex_1 to Forex_4 (4 features)
            row_values.extend([np.random.uniform(70, 85) for _ in range(4)])
            # Macro_1 to Macro_6 (6 features) - simplified, will be forward-filled in actual data
            row_values.extend([monthly_repo_rate, monthly_inflation, np.random.uniform(0.01, 0.05), np.random.uniform(0.02, 0.06), np.random.uniform(50, 60), np.random.uniform(50, 60)])
            # InstFlow_1 to InstFlow_4 (4 features)
            row_values.extend([daily_inst_flow, daily_inst_flow * 0.5, np.random.uniform(100, 500), np.random.uniform(10, 50)])
            # Sector_1 to Sector_10 (10 features)
            row_values.extend([np.random.uniform(1000, 5000) for _ in range(10)])
            # Event Flags (5 features)
            row_values.extend([is_budget_day, is_rbi_policy_day, is_geopolitical_event, is_us_fed_day, is_election_day])

            data_rows.append([timestamp] + row_values)

            # Update base price for next candle
            base_price = close_price

        current_date += timedelta(days=1)

    df = pd.DataFrame(data_rows, columns=['Timestamp'] + feature_names)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dummy data saved to {output_path}")
    print(f"DataFrame head:\n{df.head()}")
    print(f"DataFrame info:\n{df.info()}")

if __name__ == "__main__":
    # Generate 500 days of dummy data for quick testing
    # For actual training, you'd generate or collect 5 years (approx 1250 trading days)
    generate_dummy_nifty_data(num_days=500)
