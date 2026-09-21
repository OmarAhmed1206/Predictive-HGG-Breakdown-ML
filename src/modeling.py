from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import math
import pickle

@dataclass
class WeibullModel:
    shape: float  # beta (shape parameter)
    scale: float  # eta (scale parameter)
    loc: float    # location parameter (usually 0)
    n_observations: int
    aic: float
    bic: float
    log_likelihood: float

def fit_weibull(inter_failure_times: np.ndarray) -> WeibullModel:
    data = inter_failure_times[~np.isnan(inter_failure_times)]
    data = data[data > 0]
    shape, loc, scale = stats.weibull_min.fit(data, floc=0)
    
    n = len(data)
    ll = np.sum(stats.weibull_min.logpdf(data, shape, loc=loc, scale=scale))
    k = 2
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    
    return WeibullModel(shape=shape, scale=scale, loc=loc, n_observations=n, aic=aic, bic=bic, log_likelihood=ll)

def fit_exponential(inter_failure_times: np.ndarray) -> dict:
    data = inter_failure_times[~np.isnan(inter_failure_times)]
    data = data[data > 0]
    loc, scale = stats.expon.fit(data, floc=0)
    rate = 1.0 / scale
    
    n = len(data)
    ll = np.sum(stats.expon.logpdf(data, loc=loc, scale=scale))
    k = 1
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    
    return {'rate': rate, 'scale': scale, 'aic': aic, 'bic': bic, 'log_likelihood': ll}

def fit_lognormal(inter_failure_times: np.ndarray) -> dict:
    data = inter_failure_times[~np.isnan(inter_failure_times)]
    data = data[data > 0]
    shape, loc, scale = stats.lognorm.fit(data, floc=0)
    mu = np.log(scale)
    sigma = shape
    
    n = len(data)
    ll = np.sum(stats.lognorm.logpdf(data, shape, loc=loc, scale=scale))
    k = 2
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    
    return {'mu': mu, 'sigma': sigma, 'aic': aic, 'bic': bic, 'log_likelihood': ll}

def weibull_survival(model: WeibullModel, t: float) -> float:
    if t <= 0: return 1.0
    return np.exp(- (t / model.scale) ** model.shape)

def weibull_failure_prob(model: WeibullModel, t: float) -> float:
    return 1.0 - weibull_survival(model, t)

def weibull_conditional_failure_prob(model: WeibullModel, t_additional: float, elapsed: float) -> float:
    s_elapsed = weibull_survival(model, elapsed)
    if s_elapsed == 0:
        return 1.0
    s_future = weibull_survival(model, elapsed + t_additional)
    return 1.0 - (s_future / s_elapsed)

def weibull_expected_ttf(model: WeibullModel) -> float:
    return model.scale * math.gamma(1.0 + 1.0 / model.shape)

def weibull_median_ttf(model: WeibullModel) -> float:
    return model.scale * (math.log(2)) ** (1.0 / model.shape)

def weibull_hazard(model: WeibullModel, t: float) -> float:
    if t <= 0: return 0.0
    return (model.shape / model.scale) * ((t / model.scale) ** (model.shape - 1))

def weibull_percentile(model: WeibullModel, p: float) -> float:
    if p >= 1.0: return float('inf')
    if p <= 0.0: return 0.0
    return model.scale * (-math.log(1.0 - p)) ** (1.0 / model.shape)

def compare_distributions(inter_failure_times: np.ndarray) -> pd.DataFrame:
    weibull = fit_weibull(inter_failure_times)
    expo = fit_exponential(inter_failure_times)
    lognorm = fit_lognormal(inter_failure_times)
    
    records = [
        {'distribution': 'Weibull', 'params_str': f"shape={weibull.shape:.4f}, scale={weibull.scale:.4f}", 'log_likelihood': weibull.log_likelihood, 'aic': weibull.aic, 'bic': weibull.bic},
        {'distribution': 'Exponential', 'params_str': f"rate={expo['rate']:.4f}, scale={expo['scale']:.4f}", 'log_likelihood': expo['log_likelihood'], 'aic': expo['aic'], 'bic': expo['bic']},
        {'distribution': 'Log-Normal', 'params_str': f"mu={lognorm['mu']:.4f}, sigma={lognorm['sigma']:.4f}", 'log_likelihood': lognorm['log_likelihood'], 'aic': lognorm['aic'], 'bic': lognorm['bic']}
    ]
    df = pd.DataFrame(records)
    df['best'] = df['aic'] == df['aic'].min()
    return df

def kaplan_meier_estimate(inter_failure_times: np.ndarray) -> pd.DataFrame:
    data = inter_failure_times[~np.isnan(inter_failure_times)]
    data = data[data >= 0]
    unique_times, counts = np.unique(data, return_counts=True)
    
    n_at_risk = len(data)
    surv = 1.0
    var_sum = 0.0
    
    records = [{'time': 0.0, 'n_at_risk': n_at_risk, 'n_events': 0, 'survival_prob': 1.0, 'ci_lower': 1.0, 'ci_upper': 1.0}]
    
    for t, c in zip(unique_times, counts):
        if n_at_risk == 0:
            break
        prob = 1.0 - c / n_at_risk
        surv *= prob
        if n_at_risk > c:
            var_sum += c / (n_at_risk * (n_at_risk - c))
        else:
            var_sum += 0
            
        se = surv * np.sqrt(var_sum) if var_sum > 0 else 0
        ci_lower = max(0.0, surv - 1.96 * se)
        ci_upper = min(1.0, surv + 1.96 * se)
        
        records.append({
            'time': float(t),
            'n_at_risk': int(n_at_risk),
            'n_events': int(c),
            'survival_prob': surv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
        n_at_risk -= c
        
    df = pd.DataFrame(records).sort_values('time').reset_index(drop=True)
    return df


# =============================================
# Markov Chain Fault Category Prediction
# =============================================

def build_transition_matrix(categories: pd.Series) -> pd.DataFrame:
    """Build a Markov Chain transition probability matrix from a sequence of fault categories.
    
    Args:
        categories: pd.Series of category strings, ordered chronologically.
        
    Returns:
        pd.DataFrame where rows are 'from' categories and columns are 'to' categories.
        Each row sums to 1.0 (transition probabilities).
    """
    cats = categories.dropna().reset_index(drop=True)
    if len(cats) < 2:
        return pd.DataFrame()
    
    all_categories = sorted(cats.unique())
    
    # Count transitions
    transition_counts = pd.DataFrame(0, index=all_categories, columns=all_categories, dtype=float)
    for i in range(len(cats) - 1):
        from_cat = cats.iloc[i]
        to_cat = cats.iloc[i + 1]
        transition_counts.loc[from_cat, to_cat] += 1
    
    # Normalize rows to get probabilities
    row_sums = transition_counts.sum(axis=1)
    transition_probs = transition_counts.div(row_sums, axis=0).fillna(0)
    
    return transition_probs


def predict_next_category(transition_matrix: pd.DataFrame, current_category: str, top_n: int = 3) -> list:
    """Predict the most likely next fault categories using the Markov transition matrix.
    
    Args:
        transition_matrix: Output of build_transition_matrix.
        current_category: The most recent fault category.
        top_n: Number of top predictions to return.
        
    Returns:
        List of dicts: [{'category': str, 'probability': float}, ...]
        Sorted by probability descending.
    """
    if transition_matrix.empty or current_category not in transition_matrix.index:
        # Fallback: use overall category frequency (uniform-ish)
        if not transition_matrix.empty:
            avg_probs = transition_matrix.mean(axis=0)
            results = [{'category': cat, 'probability': prob} for cat, prob in avg_probs.items()]
            results.sort(key=lambda x: x['probability'], reverse=True)
            return results[:top_n]
        return []
    
    row = transition_matrix.loc[current_category]
    results = [{'category': cat, 'probability': prob} for cat, prob in row.items() if prob > 0]
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results[:top_n]


def get_category_overall_distribution(categories: pd.Series) -> dict:
    """Get the overall historical distribution of fault categories.
    
    Returns:
        Dict mapping category name to proportion (0-1).
    """
    counts = categories.value_counts(normalize=True)
    return counts.to_dict()


def save_model(model: WeibullModel, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str) -> WeibullModel:
    with open(filepath, 'rb') as f:
        return pickle.load(f)
