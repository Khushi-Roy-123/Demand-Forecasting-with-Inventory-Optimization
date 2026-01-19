import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

def run_pipeline():
    print("Starting pipeline...")
    
    # 1. Load Data
    print("Loading data...")
    dtypes = {
        'id': 'int32',
        'item_nbr': 'int32',
        'store_nbr': 'int8',
        'unit_sales': 'float32',
        'onpromotion': 'object'
    }
    
    data = {}
    if 'train.csv' in os.listdir('data'):
        # Using 1M rows for demo speed. 
        # Note: 1M rows might cover only a short period (e.g. Jan-Mar 2013)
        data['train'] = pd.read_csv(os.path.join('data', 'train.csv'), dtype=dtypes, parse_dates=['date'], nrows=1000000)
    else:
        print("Error: train.csv not found!")
        return

    if 'test.csv' in os.listdir():
        data['test'] = pd.read_csv('test.csv', dtype=dtypes, parse_dates=['date'])
    
    # Load supporting files
    for f in ['items.csv', 'stores.csv', 'oil.csv', 'holidays_events.csv']:
        if f in os.listdir('data'):
            print(f"Loading {f}...")
            name = f.split('.')[0].replace('_events','')
            file_path = os.path.join('data', f)
            if f in ['oil.csv', 'holidays_events.csv']:
                 data[name] = pd.read_csv(file_path, parse_dates=['date'])
            else:
                 data[name] = pd.read_csv(file_path)

    # 2. Preprocess
    print("Preprocessing...")
    train = data['train']
    test = data.get('test')
    
    # Merge logic
    oil = data.get('oil')
    if oil is not None:
        try:
            date_range = pd.date_range(start=oil['date'].min(), end=oil['date'].max())
            oil = oil.set_index('date').reindex(date_range).reset_index()
            oil.rename(columns={'index': 'date', 'dcoilwtico': 'oil_price'}, inplace=True)
            oil['oil_price'] = oil['oil_price'].interpolate(method='linear').fillna(method='bfill')
            train = train.merge(oil, on='date', how='left')
            if test is not None: test = test.merge(oil, on='date', how='left')
        except Exception as e:
            print(f"Oil merge failed: {e}")

    stores = data.get('stores')
    if stores is not None:
        train = train.merge(stores, on='store_nbr', how='left')
        if test is not None: test = test.merge(stores, on='store_nbr', how='left')
        
    items = data.get('items')
    if items is not None:
        train = train.merge(items, on='item_nbr', how='left')
        if test is not None: test = test.merge(items, on='item_nbr', how='left')

    holidays = data.get('holidays')
    if holidays is not None:
        try:
            holidays = holidays[holidays['transferred'] == False]
            holiday_dates = set(holidays['date'])
            train['is_holiday'] = train['date'].apply(lambda x: 1 if x in holiday_dates else 0)
            if test is not None: test['is_holiday'] = test['date'].apply(lambda x: 1 if x in holiday_dates else 0)
        except Exception:
            pass

    train['onpromotion'] = train['onpromotion'].fillna(False).astype(bool)
    if test is not None: test['onpromotion'] = test['onpromotion'].fillna(False).astype(bool)

    # 3. Feature Engineering
    print("Feature Engineering...")
    for df in [train, test]:
        if df is not None:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # 4. Train Model
    print("Preparing Training Data...")
    drop_cols = ['id', 'date', 'unit_sales', 'description', 'locale', 'locale_name', 'type_x', 'type_y', 'city', 'state']
    
    # Encode strings
    for col in train.select_dtypes(include=['object']).columns:
        if col not in drop_cols:
            le = LabelEncoder()
            train[col] = train[col].astype(str)
            train[col] = le.fit_transform(train[col])
            if test is not None and col in test.columns:
                 test[col] = test[col].astype(str)
                 # Fit transform separately for demo robustness
                 test[col] = LabelEncoder().fit_transform(test[col])

    features = [c for c in train.columns if c not in drop_cols and c != 'unit_sales']
    target = 'unit_sales'
    
    # SORT by date for time series split
    train = train.sort_values('date')
    
    # Robust Split: Use last 10% or just last 28 days if data covers enough range
    # Given 1M rows demo, safe to use simple index split to ensure non-empty sets
    split_idx = int(len(train) * 0.9)
    if split_idx == 0 or split_idx == len(train):
        print("Data too small for split. Using last 100 rows as val.")
        split_idx = max(0, len(train) - 100)
    
    X_train = train.iloc[:split_idx][features].fillna(0)
    y_train = train.iloc[:split_idx][target]
    X_val = train.iloc[split_idx:][features].fillna(0)
    y_val = train.iloc[split_idx:][target]
    
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")
    
    if len(X_train) == 0:
        print("Error: Train set empty!")
        return

    print("Training LightGBM...")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    model = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=[lgb_val], callbacks=[lgb.early_stopping(50)])
    
    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    with open(os.path.join('models', 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to models/model.pkl")
    
    # 5. Submission
    if test is not None:
        print("Generating Submission...")
        X_test = test[features].fillna(0)
        preds = model.predict(X_test)
        
        submission = pd.DataFrame({
            'id': test['id'],
            'unit_sales': preds
        })
        submission['unit_sales'] = submission['unit_sales'].clip(lower=0)
        submission.to_csv('submission.csv', index=False)
        print("submission.csv created.")

if __name__ == "__main__":
    run_pipeline()
