from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.modeling import (
    WeibullModel,
    weibull_expected_ttf, weibull_median_ttf, 
    weibull_percentile, weibull_conditional_failure_prob,
    weibull_survival, weibull_failure_prob, weibull_hazard,
    build_transition_matrix, predict_next_category
)

def predict_next_breakdown(model, last_breakdown: datetime, current_time: datetime = None) -> dict:
    if current_time is None:
        current_time = datetime.now()
        
    elapsed_days = (current_time - last_breakdown).total_seconds() / 86400.0
    
    exp_ttf = weibull_expected_ttf(model)
    med_ttf = weibull_median_ttf(model)
    
    p25 = weibull_percentile(model, 0.25)
    p75 = weibull_percentile(model, 0.75)
    p05 = weibull_percentile(model, 0.05)
    p95 = weibull_percentile(model, 0.95)
    
    return {
        'last_breakdown': last_breakdown,
        'current_time': current_time,
        'elapsed_days': elapsed_days,
        'expected_ttf_days': exp_ttf,
        'median_ttf_days': med_ttf,
        'predicted_date_expected': last_breakdown + timedelta(days=exp_ttf) if not pd.isna(exp_ttf) else None,
        'predicted_date_median': last_breakdown + timedelta(days=med_ttf) if not pd.isna(med_ttf) else None,
        'remaining_expected_days': max(0.0, exp_ttf - elapsed_days),
        'remaining_median_days': max(0.0, med_ttf - elapsed_days),
        'ci_25_days': p25,
        'ci_75_days': p75,
        'ci_5_days': p05,
        'ci_95_days': p95
    }

def risk_windows(model, elapsed_days: float, windows: list = [1, 3, 7, 14, 30]) -> dict:
    return {w: weibull_conditional_failure_prob(model, w, elapsed_days) for w in windows}

def current_risk_level(probability_7d: float) -> tuple:
    if probability_7d < 0.25:
        return ('Low', '#2ecc71')
    elif probability_7d < 0.50:
        return ('Medium', '#f39c12')
    elif probability_7d < 0.75:
        return ('High', '#e74c3c')
    else:
        return ('Critical', '#8e44ad')

def generate_survival_curve(model, max_days: float = 90, n_points: int = 200) -> pd.DataFrame:
    days = [i * max_days / (n_points - 1) for i in range(n_points)]
    records = []
    
    for d in days:
        records.append({
            'days': d,
            'survival_prob': weibull_survival(model, d),
            'failure_prob': weibull_failure_prob(model, d),
            'hazard_rate': weibull_hazard(model, d)
        })
        
    return pd.DataFrame(records)

def generate_conditional_survival_curve(model, elapsed_days: float, max_additional_days: float = 90, n_points: int = 200) -> pd.DataFrame:
    add_days = [i * max_additional_days / (n_points - 1) for i in range(n_points)]
    records = []
    
    s_elapsed = weibull_survival(model, elapsed_days)
    
    for t in add_days:
        s_future = weibull_survival(model, elapsed_days + t)
        prob_surv = s_future / s_elapsed if s_elapsed > 0 else 0.0
        
        records.append({
            'additional_days': t,
            'survival_prob': prob_surv,
            'failure_prob': 1.0 - prob_surv
        })
        
    return pd.DataFrame(records)


# =============================================
# Forward-Chaining Breakdown Schedule Simulator
# =============================================

def simulate_future_breakdowns(
    model: WeibullModel,
    transition_matrix: pd.DataFrame,
    last_breakdown_date: datetime,
    last_category: str,
    horizon_days: int = 365,
    use_median: bool = True
) -> pd.DataFrame:
    """Simulate a forward-chaining schedule of predicted breakdowns.
    
    Starting from the last known breakdown, the simulator repeatedly:
    1. Computes the expected/median TTF from the Weibull model
    2. Adds that to the current date to get the next predicted breakdown date
    3. Uses the Markov transition matrix to predict the most likely fault category
    4. Repeats until the horizon is exceeded
    
    Args:
        model: Fitted WeibullModel
        transition_matrix: Markov chain transition matrix (from build_transition_matrix)
        last_breakdown_date: datetime of the most recent known breakdown
        last_category: category string of the most recent breakdown
        horizon_days: how many days into the future to simulate (default 365)
        use_median: if True, use median TTF; if False, use expected (mean) TTF
    
    Returns:
        pd.DataFrame with predicted schedule columns.
    """
    ttf = weibull_median_ttf(model) if use_median else weibull_expected_ttf(model)
    
    if ttf <= 0 or pd.isna(ttf):
        return pd.DataFrame()
    
    current_date = last_breakdown_date
    current_category = last_category
    now = datetime.now()
    
    records = []
    event_num = 0
    
    # Confidence range via Weibull percentiles
    p25_ttf = weibull_percentile(model, 0.25)
    p75_ttf = weibull_percentile(model, 0.75)
    
    # Single RNG seeded once for the entire simulation — each loop iteration
    # advances the internal state, producing genuinely different draws.
    rng = np.random.RandomState(42)
    
    while True:
        event_num += 1
        next_date = current_date + timedelta(days=ttf)
        
        # Stop if we exceed the horizon from NOW
        if (next_date - now).total_seconds() / 86400 > horizon_days:
            break
        
        # Get full transition row for current_category
        if current_category in transition_matrix.index:
            row = transition_matrix.loc[current_category]
        else:
            # Fallback: use column-wise mean (overall distribution)
            row = transition_matrix.mean(axis=0)
        
        cats = row.index.tolist()
        probs = row.values.astype(float)
        probs = probs / probs.sum()  # normalise
        
        # Sample from the distribution
        predicted_cat = rng.choice(cats, p=probs)
        predicted_prob = float(row[predicted_cat])
        
        # Top 3 for display (always sorted by probability)
        sorted_idx = np.argsort(-probs)
        top_3 = [(cats[i], probs[i]) for i in sorted_idx[:3]]
        top3_str = ' | '.join([f"{c}: {p*100:.0f}%" for c, p in top_3])
        
        days_from_now = (next_date - now).total_seconds() / 86400
        
        records.append({
            'event_number': event_num,
            'predicted_date': next_date,
            'days_from_now': round(days_from_now, 1),
            'predicted_category': predicted_cat,
            'category_probability': round(predicted_prob * 100, 1),
            'top_3_categories': top3_str,
            'earliest_estimate': current_date + timedelta(days=p25_ttf),
            'latest_estimate': current_date + timedelta(days=p75_ttf),
        })
        
        # Advance for next iteration
        current_date = next_date
        current_category = predicted_cat
        
        # Safety: max 50 events
        if event_num >= 50:
            break
    
    return pd.DataFrame(records)
