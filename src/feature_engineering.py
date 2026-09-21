import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['inter_failure_days'] = df['actual_start'].diff().dt.total_seconds() / (24 * 3600.0)
    
    df['day_of_week'] = df['actual_start'].dt.dayofweek
    df['month'] = df['actual_start'].dt.month
    df['year'] = df['actual_start'].dt.year
    df['hour'] = df['actual_start'].dt.hour
    
    # rolling counts
    df_indexed = df.set_index('actual_start').sort_index()
    
    rolling_30 = df_indexed.rolling('30D').count()['wo_no'].values
    rolling_60 = df_indexed.rolling('60D').count()['wo_no'].values
    rolling_90 = df_indexed.rolling('90D').count()['wo_no'].values
    
    df['rolling_count_30d'] = rolling_30
    df['rolling_count_60d'] = rolling_60
    df['rolling_count_90d'] = rolling_90
    
    df['cumulative_count'] = range(1, len(df) + 1)
    
    df['prev_work_type'] = df['work_type'].shift(1)
    df['prev_duration'] = df['duration_hours'].shift(1)
    
    return df

def compute_mtbf_trend(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    res = pd.DataFrame()
    res['event_index'] = df.index
    res['actual_start'] = df['actual_start']
    res['inter_failure_days'] = df['inter_failure_days']
    res['mtbf_rolling'] = res['inter_failure_days'].rolling(window=window).mean()
    return res

def compute_rolling_failure_rate(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    min_date = df['actual_start'].min().floor('D')
    max_date = df['actual_start'].max().ceil('D')
    
    if pd.isna(min_date) or pd.isna(max_date):
        return pd.DataFrame(columns=['date', 'failure_count', 'failure_rate'])
        
    dates = pd.date_range(min_date, max_date, freq='D')
    
    counts = []
    rates = []
    
    for d in dates:
        start_window = d - pd.Timedelta(days=window_days)
        events_in_window = df[(df['actual_start'] >= start_window) & (df['actual_start'] <= d)]
        c = len(events_in_window)
        counts.append(c)
        rates.append(c / window_days)
        
    res = pd.DataFrame({
        'date': dates,
        'failure_count': counts,
        'failure_rate': rates
    })
    return res
