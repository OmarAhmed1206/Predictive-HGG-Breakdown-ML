import numpy as np
import pandas as pd
from src.modeling import fit_weibull, weibull_median_ttf, weibull_failure_prob, weibull_percentile, weibull_expected_ttf

def chronological_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple:
    n = len(df)
    train_size = int(n * train_frac)
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()

def evaluate_weibull_predictions(model, test_inter_failure_times: np.ndarray) -> dict:
    actuals = test_inter_failure_times[~np.isnan(test_inter_failure_times)]
    if len(actuals) == 0:
        return {}
    
    pred = weibull_median_ttf(model)
    errors = actuals - pred
    abs_errors = np.abs(errors)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    median_ae = np.median(abs_errors)
    
    mape = np.mean(abs_errors / actuals) if np.all(actuals > 0) else None
    
    p5 = weibull_percentile(model, 0.05)
    p25 = weibull_percentile(model, 0.25)
    p75 = weibull_percentile(model, 0.75)
    p95 = weibull_percentile(model, 0.95)
    
    within_50ci = np.mean((actuals >= p25) & (actuals <= p75)) * 100.0
    within_90ci = np.mean((actuals >= p5) & (actuals <= p95)) * 100.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'median_ae': median_ae,
        'mape': mape,
        'within_50ci': within_50ci,
        'within_90ci': within_90ci
    }

def baseline_metrics(train_ift: np.ndarray, test_ift: np.ndarray) -> dict:
    train_clean = train_ift[~np.isnan(train_ift)]
    test_clean = test_ift[~np.isnan(test_ift)]
    
    if len(train_clean) == 0 or len(test_clean) == 0:
        return {}
        
    mean_val = np.mean(train_clean)
    median_val = np.median(train_clean)
    
    mae_mean = np.mean(np.abs(test_clean - mean_val))
    rmse_mean = np.sqrt(np.mean((test_clean - mean_val)**2))
    
    mae_median = np.mean(np.abs(test_clean - median_val))
    rmse_median = np.sqrt(np.mean((test_clean - median_val)**2))
    
    return {
        'mae_mean': mae_mean,
        'rmse_mean': rmse_mean,
        'mae_median': mae_median,
        'rmse_median': rmse_median
    }

def rolling_validation(inter_failure_times: np.ndarray, min_train: int = 30) -> pd.DataFrame:
    data = inter_failure_times[~np.isnan(inter_failure_times)]
    records = []
    
    for k in range(min_train, len(data)):
        train_data = data[:k]
        actual = data[k]
        
        try:
            model = fit_weibull(train_data)
            pred_median = weibull_median_ttf(model)
            pred_mean = weibull_expected_ttf(model)
        except Exception:
            pred_median = np.nan
            pred_mean = np.nan
            
        err = actual - pred_median
        abs_err = abs(err)
        
        records.append({
            'train_size': k,
            'actual_ift': actual,
            'predicted_median': pred_median,
            'predicted_mean': pred_mean,
            'error': err,
            'abs_error': abs_err
        })
        
    return pd.DataFrame(records)

def brier_score_windows(model, test_ift: np.ndarray, windows: list = [1, 3, 7, 14, 30]) -> dict:
    actuals = test_ift[~np.isnan(test_ift)]
    results = {}
    if len(actuals) == 0:
        return results
        
    for w in windows:
        outcomes = (actuals <= w).astype(float)
        prob = weibull_failure_prob(model, w)
        brier = np.mean((prob - outcomes)**2)
        results[w] = brier
        
    return results

def compute_concordance_index(predicted_risks: np.ndarray, actual_times: np.ndarray) -> float:
    n = len(actual_times)
    concordant = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i+1, n):
            if actual_times[i] != actual_times[j]:
                total_pairs += 1
                if actual_times[i] < actual_times[j] and predicted_risks[i] > predicted_risks[j]:
                    concordant += 1
                elif actual_times[i] > actual_times[j] and predicted_risks[i] < predicted_risks[j]:
                    concordant += 1
                    
    if total_pairs == 0:
        return 0.5
    return concordant / total_pairs
