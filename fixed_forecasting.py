import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """Load and prepare the CSV data for forecasting"""
    # Load the data
    df = pd.read_csv(file_path)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
    
    # Sort by date
    df = df.sort_values('DATE')
    
    # Create time features
    df['month'] = df['DATE'].dt.month
    df['year'] = df['DATE'].dt.year
    df['quarter'] = df['DATE'].dt.quarter
    df['day_of_year'] = df['DATE'].dt.dayofyear
    
    # Create a sequential time index for each lineup separately
    df['time_index'] = df.groupby('Lineup').cumcount()
    
    return df

def create_advanced_features(df):
    """Create advanced features for the forecasting model"""
    # Create lag features (previous month values) - only for training, not for the complete dataset
    df['lag_1'] = df.groupby('Lineup')['Actual'].shift(1)
    df['lag_2'] = df.groupby('Lineup')['Actual'].shift(2)
    df['lag_3'] = df.groupby('Lineup')['Actual'].shift(3)
    df['lag_6'] = df.groupby('Lineup')['Actual'].shift(6)
    df['lag_12'] = df.groupby('Lineup')['Actual'].shift(12)
    
    # Create rolling statistics
    df['rolling_mean_3'] = df.groupby('Lineup')['Actual'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_mean_6'] = df.groupby('Lineup')['Actual'].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_std_3'] = df.groupby('Lineup')['Actual'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True)
    df['rolling_std_6'] = df.groupby('Lineup')['Actual'].rolling(window=6, min_periods=1).std().reset_index(0, drop=True)
    
    # Create seasonal features
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['sin_quarter'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['cos_quarter'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # Create trend features
    df['trend'] = df['time_index']
    df['trend_squared'] = df['time_index'] ** 2
    
    # Create interaction features
    df['month_trend'] = df['month'] * df['trend']
    df['quarter_trend'] = df['quarter'] * df['trend']
    
    # Create difference features
    df['diff_1'] = df.groupby('Lineup')['Actual'].diff(1)
    df['diff_2'] = df.groupby('Lineup')['Actual'].diff(2)
    
    # Create percentage change features
    df['pct_change_1'] = df.groupby('Lineup')['Actual'].pct_change(1)
    df['pct_change_3'] = df.groupby('Lineup')['Actual'].pct_change(3)
    
    # Fill NaN values with appropriate defaults
    df = df.fillna({
        'lag_1': df.groupby('Lineup')['Actual'].transform('mean'),
        'lag_2': df.groupby('Lineup')['Actual'].transform('mean'),
        'lag_3': df.groupby('Lineup')['Actual'].transform('mean'),
        'lag_6': df.groupby('Lineup')['Actual'].transform('mean'),
        'lag_12': df.groupby('Lineup')['Actual'].transform('mean'),
        'rolling_mean_3': df.groupby('Lineup')['Actual'].transform('mean'),
        'rolling_mean_6': df.groupby('Lineup')['Actual'].transform('mean'),
        'rolling_std_3': 0,
        'rolling_std_6': 0,
        'diff_1': 0,
        'diff_2': 0,
        'pct_change_1': 0,
        'pct_change_3': 0
    })
    
    return df

def train_forecasting_model(df, lineup_value):
    """Train a forecasting model for a specific lineup"""
    # Filter data for specific lineup
    lineup_data = df[df['Lineup'] == lineup_value].copy()
    
    if len(lineup_data) < 15:
        print(f"Warning: Insufficient data for lineup {lineup_value}")
        return None, None, None, None, None
    
    # Prepare features and target
    feature_columns = ['time_index', 'month', 'year', 'quarter', 'day_of_year',
                      'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                      'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'rolling_std_6',
                      'sin_month', 'cos_month', 'sin_quarter', 'cos_quarter',
                      'trend', 'trend_squared', 'month_trend', 'quarter_trend',
                      'diff_1', 'diff_2', 'pct_change_1', 'pct_change_3']
    
    X = lineup_data[feature_columns]
    y = lineup_data['Actual']
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, mape, mae, r2

def forecast_future_months(model, df, lineup_value, months_ahead=12):
    """Forecast the next N months"""
    # Get the last data point for this lineup
    last_data = df[df['Lineup'] == lineup_value].iloc[-1]
    
    # Create future dates - ensure they align with the 1st of each month
    last_date = last_data['DATE']
    future_dates = []
    current_date = last_date
    
    for i in range(1, months_ahead + 1):
        # Calculate the next month
        if current_date.month == 12:
            next_month = 1
            next_year = current_date.year + 1
        else:
            next_month = current_date.month + 1
            next_year = current_date.year
        
        # Create date for the 1st of the next month
        future_date = datetime(next_year, next_month, 1)
        future_dates.append(future_date)
        current_date = future_date
    
    # Create future features
    future_features = []
    for i, date in enumerate(future_dates):
        month = date.month
        year = date.year
        quarter = (month - 1) // 3 + 1
        day_of_year = date.timetuple().tm_yday
        time_index = last_data['time_index'] + i + 1
        
        # Use the last known values for lag features
        recent_data = df[df['Lineup'] == lineup_value].tail(12)
        lag_1 = last_data['Actual']
        lag_2 = recent_data.iloc[-2]['Actual'] if len(recent_data) > 1 else lag_1
        lag_3 = recent_data.iloc[-3]['Actual'] if len(recent_data) > 2 else lag_2
        lag_6 = recent_data.iloc[-6]['Actual'] if len(recent_data) > 5 else lag_1
        lag_12 = recent_data.iloc[-12]['Actual'] if len(recent_data) > 11 else lag_1
        
        # Calculate rolling statistics
        recent_values = recent_data['Actual'].values
        rolling_mean_3 = np.mean(recent_values[-3:]) if len(recent_values) >= 3 else lag_1
        rolling_mean_6 = np.mean(recent_values) if len(recent_values) >= 6 else rolling_mean_3
        rolling_std_3 = np.std(recent_values[-3:]) if len(recent_values) >= 3 else 0
        rolling_std_6 = np.std(recent_values) if len(recent_values) >= 6 else 0
        
        # Seasonal features
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)
        sin_quarter = np.sin(2 * np.pi * quarter / 4)
        cos_quarter = np.cos(2 * np.pi * quarter / 4)
        
        # Trend features
        trend = time_index
        trend_squared = time_index ** 2
        
        # Interaction features
        month_trend = month * trend
        quarter_trend = quarter * trend
        
        # Difference features (use recent differences)
        diff_1 = lag_1 - lag_2
        diff_2 = lag_2 - lag_3
        
        # Percentage change features
        pct_change_1 = (lag_1 - lag_2) / lag_2 if lag_2 != 0 else 0
        pct_change_3 = (lag_1 - lag_3) / lag_3 if lag_3 != 0 else 0
        
        features = [time_index, month, year, quarter, day_of_year,
                   lag_1, lag_2, lag_3, lag_6, lag_12,
                   rolling_mean_3, rolling_mean_6, rolling_std_3, rolling_std_6,
                   sin_month, cos_month, sin_quarter, cos_quarter,
                   trend, trend_squared, month_trend, quarter_trend,
                   diff_1, diff_2, pct_change_1, pct_change_3]
        future_features.append(features)
    
    # Make predictions
    future_predictions = model.predict(future_features)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_predictions
    })
    
    return forecast_df

def main():
    """Main function to run the fixed forecasting analysis"""
    print("Loading and preparing data...")
    df = load_and_prepare_data('Sample_data_N.csv')
    
    # Load plan numbers for forecasted period
    print("Loading plan numbers for forecasted period...")
    plan_df = pd.read_csv('Plan Number.csv')
    plan_df['DATE'] = pd.to_datetime(plan_df['DATE'], format='%d-%m-%Y')
    
    # Get unique lineups
    lineups = df['Lineup'].unique()
    print(f"Found {len(lineups)} lineups: {lineups}")
    
    # Print initial data counts
    for lineup in lineups:
        lineup_count = len(df[df['Lineup'] == lineup])
        print(f"Lineup {lineup}: {lineup_count} records")
    
    # Prepare features (this should preserve all records now)
    df = create_advanced_features(df)
    
    # Print data counts after feature engineering
    print(f"\nAfter feature engineering:")
    for lineup in lineups:
        lineup_count = len(df[df['Lineup'] == lineup])
        print(f"Lineup {lineup}: {lineup_count} records")
    
    results = {}
    
    for lineup in lineups:
        print(f"\n{'='*50}")
        print(f"Analyzing Lineup: {lineup}")
        print(f"{'='*50}")
        
        # Train model and get predictions
        result = train_forecasting_model(df, lineup)
        
        if result is None:
            continue
            
        model, X_test, y_test, y_pred, mape, mae, r2 = result
        
        print(f"Model trained successfully!")
        print(f"Test MAPE: {mape:.2f}%")
        print(f"Test MAE: £{mae:.2f}")
        print(f"Test R²: {r2:.4f}")
        
        # Forecast next 12 months
        forecast_df = forecast_future_months(model, df, lineup, months_ahead=12)
        
        print(f"\nForecast for next 12 months:")
        print(forecast_df.to_string(index=False))
        
        # Store results
        results[lineup] = {
            'model': model,
            'mape': mape,
            'mae': mae,
            'r2': r2,
            'forecast': forecast_df
        }
    
    # Create complete dataframe with original data + forecasts
    print(f"\nCreating complete dataframe with original data + forecasts...")
    
    # Start with the original data (all 72 records)
    complete_df = df.copy()
    
    # Add forecast data for each lineup
    for lineup, result in results.items():
        # Get the forecast data for this lineup
        lineup_forecast = result['forecast'].copy()
        
        # Create forecast rows with the same structure as original data
        forecast_rows = []
        for _, forecast_row in lineup_forecast.iterrows():
            # Get the corresponding plan number for this date and lineup
            forecast_date = forecast_row['Date']
            # Convert forecast date to the same format as plan dates for matching
            forecast_date_str = forecast_date.strftime('%d-%m-%Y')
            plan_value = plan_df[(plan_df['DATE'].dt.strftime('%d-%m-%Y') == forecast_date_str) & (plan_df['Lineup'] == lineup)]['Plan'].values
            
            # Use plan value if found, otherwise use NaN
            plan_number = plan_value[0] if len(plan_value) > 0 else np.nan
            
            # Create a new row with the same structure as original data
            new_row = {
                'Profile': 'asm23',
                'Line_Item': 'Action',
                'Budget Unit': 'UBC01',
                'Token': 'GBP',
                'Body': 'AWS',
                'Site': 'LBS',
                'Lineup': lineup,
                'Institutions': 'Bank',
                'DATE': forecast_row['Date'].strftime('%d-%m-%Y'),
                'Actual': np.nan,
                'Plan': plan_number,
                'Forecast': forecast_row['Forecast']
            }
            forecast_rows.append(new_row)
        
        # Convert forecast rows to DataFrame
        forecast_df = pd.DataFrame(forecast_rows)
        
        # Append forecast data to the complete dataframe
        complete_df = pd.concat([complete_df, forecast_df], ignore_index=True)
    
    # Sort the complete dataframe by Lineup and DATE
    complete_df['DATE'] = pd.to_datetime(complete_df['DATE'], format='%d-%m-%Y')
    complete_df = complete_df.sort_values(['Lineup', 'DATE'])
    
    # Save the complete dataframe
    complete_df.to_csv('fixed_complete_data_with_forecasts.csv', index=False)
    print(f"Complete dataframe saved to 'fixed_complete_data_with_forecasts.csv'")
    
    # Print summary of the complete dataframe
    print(f"\nComplete dataframe summary:")
    print(f"Total rows: {len(complete_df)}")
    print(f"Original data rows: {len(df)}")
    print(f"Forecast rows: {len(complete_df) - len(df)}")
    print(f"Lineups in complete data: {complete_df['Lineup'].unique()}")
    
    # Verify counts for each lineup
    for lineup in lineups:
        lineup_total = len(complete_df[complete_df['Lineup'] == lineup])
        lineup_actual = len(complete_df[(complete_df['Lineup'] == lineup) & (complete_df['Actual'].notna())])
        lineup_forecast = len(complete_df[(complete_df['Lineup'] == lineup) & (complete_df['Forecast'].notna())])
        lineup_plan = len(complete_df[(complete_df['Lineup'] == lineup) & (complete_df['Plan'].notna())])
        print(f"Lineup {lineup}: {lineup_total} total rows ({lineup_actual} actual + {lineup_forecast} forecast + {lineup_plan} with plan)")
    
    # Create clean version with only essential columns
    essential_columns = [
        'Profile', 'Line_Item', 'Budget Unit', 'Token', 'Body', 'Site', 
        'Lineup', 'Institutions', 'DATE', 'Actual', 'Plan', 'Forecast'
    ]
    
    clean_df = complete_df[essential_columns].copy()
    clean_df.to_csv('fixed_clean_complete_data_with_forecasts.csv', index=False)
    print(f"\nClean complete dataframe saved to 'fixed_clean_complete_data_with_forecasts.csv'")
    
    # Print summary
    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*50}")
    
    for lineup, result in results.items():
        print(f"Lineup {lineup}:")
        print(f"  MAPE: {result['mape']:.2f}%")
        print(f"  MAE: £{result['mae']:.2f}")
        print(f"  R²: {result['r2']:.4f}")
        avg_forecast = result['forecast']['Forecast'].mean()
        print(f"  Average 12-month forecast: £{avg_forecast:.2f}")
        print()

if __name__ == "__main__":
    main()
