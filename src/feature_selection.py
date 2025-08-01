import optuna
import numpy as np
from scipy.stats import spearmanr
import joblib
try:
    import cupy as cp
except ImportError:
    cp = None

def band_ratio(X, n1, n2, n3, n4, a, b):
    """
    Evaluate a generic band ratio.
    """
    if cp is not None and isinstance(X, cp.ndarray):
        band_ratio = (X[..., n1] + a * X[..., n2]) / (X[..., n3] + b * X[..., n4])
        band_ratio = cp.nan_to_num(band_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        return band_ratio.get()  # Convert back to numpy
    else:
        band_ratio = (X[..., n1] + a * X[..., n2]) / (X[..., n3] + b * X[..., n4])
        band_ratio = np.nan_to_num(band_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        return band_ratio

def objective(trial, train_data, y, sensor, class_idx):
    """
    Optuna objective function for band ratio optimization.
    """
    X = train_data[sensor]
    n = X.shape[-1] - 1

    n1 = trial.suggest_int('n1', 0, n)
    n2 = trial.suggest_int('n2', 0, n)
    n3 = trial.suggest_int('n3', 0, n)
    n4 = trial.suggest_int('n4', 0, n)

    a = trial.suggest_categorical('a', [-1, 0, 1])
    b = trial.suggest_categorical('b', [-1, 0, 1])

    br = band_ratio(X, n1, n2, n3, n4, a, b)
    r, _ = spearmanr(br, y[:, class_idx])
    
    if not np.isfinite(r):
        return 0
    
    return r

def optimize_band_ratios(train_data, test_data, y_train, y_test, classes, row_names, n_trials=2000):
    """
    Optimize band ratios for each class using Optuna.
    """
    out = {}
    for c in classes:
        print(f"Fitting band ratio for {c}...", end='')
        class_idx = row_names.index(c)
        y = y_train[:, class_idx]

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(consider_prior=False, n_startup_trials=1000)
        )
        for sensor in ['sensor1', 'sensor2', 'sensor3']:
            study.optimize(
                lambda trial: objective(trial, train_data, y_train, sensor, class_idx),
                n_trials=n_trials // 3,
                n_jobs=1,
                gc_after_trial=True
            )

        r = study.best_value
        out[c] = (study.best_trial.params, r)
        
        joblib.dump(out, f'BandRatiosTPE_Spearman_{c}.joblib')
        print(f"Done (R2=%.3f)" % r)
    
    # Compute test scores
    test_scores = {}
    for c in classes:
        args, train_score = out[c]
        args = args.copy()
        sensor = args.pop('sensor')
        br = band_ratio(test_data[sensor], **args)
        test_score = spearmanr(y_test[:, row_names.index(c)], br)[0]
        test_scores[c] = test_score
        print(f"{c}: test_r2 = %.3f train_r2 = %.3f" % (c, test_score, train_score))
    
    return out, test_scores

def apply_band_ratio(study, box, sensor):
    """
    Apply optimized features / parameters.
    """
    params = study.best_trial.params
    n1 = params['n1']
    n2 = params['n2']
    n3 = params['n3']
    n4 = params['n4']
    a = params['a']
    b = params['b']
    
    image = box.get(sensor)
    image.mask(box.mask.data[..., 0] == 0)
    
    band_ratio = (image.data[..., n1] + a * image.data[..., n2]) / (image.data[..., n3] + b * image.data[..., n4])
    band_ratio = np.nan_to_num(band_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    return hylite.HyImage(band_ratio)