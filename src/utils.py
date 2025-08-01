import numpy as np
import pickle
import os
import csv

def mean_without_zeros(array):
    """
    Calculate the mean of a numpy array, excluding zero values.
    """
    non_zero_values = array[array != 0]
    if len(non_zero_values) == 0:
        return 0
    return np.mean(non_zero_values)

def save_results(results_path, best_model=None, best_trials=None, global_top_indices=None, global_feature_mappings=None, predictions=None, band_ratios=None):
    """
    Save model, trial information, indices, predictions, or band ratio results.
    """
    os.makedirs(results_path, exist_ok=True)
    
    if best_model:
        with open(os.path.join(results_path, 'best_model_.pkl'), 'wb') as file:
            pickle.dump(best_model, file)
    
    if best_trials:
        best_trial_info = [{
            "trial_number": trial.number,
            "r_squared": trial.values[0],
            "rmse": trial.values[1],
            "ssim": trial.values[2]
        } for trial in best_trials]
        with open(os.path.join(results_path, 'best_trial_.pkl'), 'wb') as file:
            pickle.dump(best_trial_info, file)
    
    if predictions is not None:
        np.save(os.path.join(results_path, 'predictions_.npy'), predictions)
    
    if global_top_indices and global_feature_mappings:
        with open(os.path.join(results_path, 'top_indices_.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Top Index', 'Associated Bands'])
            for i, idx in enumerate(global_top_indices):
                bands_for_feature = global_feature_mappings[idx]
                writer.writerow([f"index{idx + 1}", ', '.join(map(str, bands_for_feature))])
    
    if band_ratios:
        with open(os.path.join(results_path, 'band_ratios.pkl'), 'wb') as file:
            pickle.dump(band_ratios, file)

def load_results(results_path):
    """
    Load saved results.
    """
    best_trial_info = None
    loaded_top_indices = []
    loaded_feature_mappings = {}
    band_ratios = None
    
    if os.path.exists(os.path.join(results_path, 'best_trial_.pkl')):
        with open(os.path.join(results_path, 'best_trial_.pkl'), 'rb') as file:
            best_trial_info = pickle.load(file)
    
    if os.path.exists(os.path.join(results_path, 'top_indices_.csv')):
        with open(os.path.join(results_path, 'top_indices_.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                top_index = int(row[0].replace('index', '')) - 1
                associated_bands = list(map(int, row[1].split(', ')))
                loaded_top_indices.append(top_index)
                loaded_feature_mappings[top_index] = associated_bands
    
    if os.path.exists(os.path.join(results_path, 'band_ratios.pkl')):
        with open(os.path.join(results_path, 'band_ratios.pkl'), 'rb') as file:
            band_ratios = pickle.load(file)
    
    return best_trial_info, loaded_top_indices, loaded_feature_mappings, band_ratios