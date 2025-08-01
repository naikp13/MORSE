import numpy as np
import joblib
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from skimage.metrics import structural_similarity as ssim

def train_and_predict(X, C_ref, global_top_indices, global_feature_mappings, abund_shape, cls=0):
    """
    Train the model with top features and generate predictions.
    """
    findex_list = []
    for idx in global_top_indices:
        bands = global_feature_mappings[idx]
        if len(bands) == 2:
            if idx == 0:
                feature = X[:, bands[0]] - X[:, bands[1]]
            elif idx == 1:
                feature = X[:, bands[0]] + X[:, bands[1]]
            elif idx == 2:
                feature = (X[:, bands[0]] - X[:, bands[1]]) / (X[:, bands[0]] + X[:, bands[1]] + 1e-11)
            else:
                feature = (X[:, bands[0]]) / (X[:, bands[1]] + 1e-11)
        elif len(bands) == 3:
            if idx == 4:
                feature = (X[:, bands[0]]) / (X[:, bands[1]] - X[:, bands[2]] + 1e-11)
            elif idx == 5:
                feature = (X[:, bands[0]]) / (X[:, bands[1]] + X[:, bands[2]] + 1e-11)
            elif idx == 6:
                feature = (X[:, bands[0]] - X[:, bands[1]]) / (X[:, bands[2]] + 1e-11)
            else:
                feature = (X[:, bands[0]] + X[:, bands[1]]) / (X[:, bands[2]] + 1e-11)
        elif len(bands) == 4:
            if idx == 8:
                feature = (X[:, bands[0]] - X[:, bands[1]]) / (X[:, bands[2]] - X[:, bands[3]] + 1e-11)
            elif idx == 9:
                feature = (X[:, bands[0]] - X[:, bands[1]]) / (X[:, bands[2]] + X[:, bands[3]] + 1e-11)
            elif idx == 10:
                feature = (X[:, bands[0]] + X[:, bands[1]]) / (X[:, bands[2]] - X[:, bands[3]] + 1e-11)
            else:
                feature = (X[:, bands[0]] + X[:, bands[1]]) / (X[:, bands[2]] + X[:, bands[3]] + 1e-11)
        findex_list.append(feature)
    
    findex = np.stack(findex_list, axis=1)
    findex[np.isnan(findex)] = 0
    
    best_model = joblib.load("best_model.pkl")
    predictions = best_model.predict(findex)
    predictions = np.clip(predictions, 0, np.max(C_ref))
    
    predicted = predictions.reshape(abund_shape[:, :, cls].shape)
    return predicted, predictions