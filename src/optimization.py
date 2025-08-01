import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from skimage.metrics import structural_similarity as ssim
import shap
import joblib

def calculate_indices(X, bands):
    """
    Calculate spectral indices based on selected bands.
    """
    index1 = X[:, bands[0]] - X[:, bands[1]]
    index2 = X[:, bands[2]] + X[:, bands[3]]
    index3 = (X[:, bands[4]] - X[:, bands[5]]) / (X[:, bands[4]] + X[:, bands[5]] + 1e-11)
    index4 = X[:, bands[6]] / (X[:, bands[7]] + 1e-11)
    index5 = X[:, bands[8]] / (X[:, bands[9]] - X[:, bands[10]] + 1e-11)
    index6 = X[:, bands[11]] / (X[:, bands[12]] + X[:, bands[13]] + 1e-11)
    index7 = (X[:, bands[14]] - X[:, bands[15]]) / (X[:, bands[16]] + 1e-11)
    index8 = (X[:, bands[17]] + X[:, bands[18]]) / (X[:, bands[19]] + 1e-11)
    index9 = (X[:, bands[20]] - X[:, bands[21]]) / (X[:, bands[22]] - X[:, bands[23]] + 1e-11)
    index10 = (X[:, bands[24]] - X[:, bands[25]]) / (X[:, bands[26]] + X[:, bands[27]] + 1e-11)
    index11 = (X[:, bands[28]] + X[:, bands[29]]) / (X[:, bands[30]] - X[:, bands[31]] + 1e-11)
    index12 = (X[:, bands[32]] + X[:, bands[33]]) / (X[:, bands[34]] + X[:, bands[35]] + 1e-11)
    return np.stack((index1, index2, index3, index4, index5, index6, index7, index8, index9, index10, index11, index12), axis=1)

def objective(trial, X, C_ref, nb=36):
    """
    Optuna objective function for multi-objective optimization.
    """
    global global_top_indices, global_top_bands, global_feature_mappings
    num_bands = X.shape[1]
    bands = [trial.suggest_int(f"B{i+1}", 0, num_bands-1) for i in range(nb)]
    pairs = [(0, 1), (4, 5), (9, 10), (14, 15), (20, 21), (22, 23), (24, 25), (30, 31)]
    for i, j in pairs:
        if bands[i] == bands[j]:
            return -float('inf'), float('inf'), -float('inf')
    
    features = calculate_indices(X, bands)
    X_train, X_test, C_train, C_test = train_test_split(features, C_ref, test_size=0.2, random_state=42)
    
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    C_train[np.isnan(C_train)] = 0
    C_test[np.isnan(C_test)] = 0
    
    sample_weight = np.zeros_like(C_train)
    sample_weight[(C_train >= 0) & (C_train < 0.1)] = 0.256 * 20
    sample_weight[(C_train >= 0.1) & (C_train < 0.2)] = 0.865 * 15
    sample_weight[(C_train >= 0.2) & (C_train < 0.3)] = 0.9463 * 15
    sample_weight[(C_train >= 0.3) & (C_train < 0.4)] = 0.9741 * 15
    sample_weight[(C_train >= 0.4) & (C_train < 0.5)] = 0.9822 * 15
    sample_weight[(C_train >= 0.5) & (C_train < 0.6)] = 0.9890 * 15
    sample_weight[(C_train >= 0.6) & (C_train < 0.7)] = 0.9957 * 15
    sample_weight[(C_train >= 0.7) & (C_train < 0.8)] = 0.9959 * 15
    sample_weight[(C_train >= 0.8) & (C_train <= 0.9)] = 0.9981 * 15
    sample_weight[(C_train >= 0.9) & (C_train <= 1)] = 0.9985 * 15
    
    model = linear_model.Ridge(alpha=0.5)
    model.fit(X_train, C_train, sample_weight=sample_weight)
    
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_shap_values)[-1:]
    
    global_top_indices = top_indices.tolist()
    feature_mappings = {
        0: (bands[0], bands[1]), 1: (bands[2], bands[3]), 2: (bands[4], bands[5]), 3: (bands[6], bands[7]),
        4: (bands[8], bands[9], bands[10]), 5: (bands[11], bands[12], bands[13]), 6: (bands[14], bands[15], bands[16]),
        7: (bands[17], bands[18], bands[19]), 8: (bands[20], bands[21], bands[22], bands[23]),
        9: (bands[24], bands[25], bands[26], bands[27]), 10: (bands[28], bands[29], bands[30], bands[31]),
        11: (bands[32], bands[33], bands[34], bands[35])
    }
    global_feature_mappings = {i: feature_mappings[i] for i in top_indices}
    
    X_train_top = X_train[:, top_indices]
    X_test_top = X_test[:, top_indices]
    model.fit(X_train_top, C_train, sample_weight=sample_weight)
    C_pred = model.predict(X_test_top)
    
    r_squared = r2_score(C_test, C_pred)
    rmse = mean_squared_error(C_test, C_pred, squared=False)
    data_range = C_test.max() - C_test.min()
    ssim_index, _ = ssim(C_test, C_pred, data_range=data_range, full=True)
    
    best_score = trial.study.user_attrs.get("best_score", -np.inf)
    if best_score < r_squared:
        trial.study.user_attrs["best_score"] = r_squared
        joblib.dump(model, "best_model.pkl")
    
    return r_squared, rmse, ssim_index

class MOEarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_values = None
        self.counter = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if trial.values is None:
            return
        if self.best_values is None:
            self.best_values = trial.values
            return
        improved = False
        for i, direction in enumerate(study.directions):
            if direction == optuna.study.StudyDirection.MAXIMIZE:
                if trial.values[i] > self.best_values[i] + self.min_delta:
                    improved = True
            else:
                if trial.values[i] < self.best_values[i] - self.min_delta:
                    improved = True
        if improved:
            self.best_values = trial.values
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print("Early stopping triggered.")
            study.stop()