from data_preprocessing import load_data, preprocess_data, resample_images
from optimization import objective, MOEarlyStopping
from model_training import train_and_predict
from visualization import plot_abundance_maps, plot_histograms, calculate_ssim
from utils import mean_without_zeros, save_results, load_results
import optuna
import numpy as np

def main():
    # Paths
    hydat_path = 'data.hdr'
    abund_path = 'ref.hdr'
    results_path = '/path/to/results/'
    outdir = '/path/to/aux/outputs/'
    
    # Load and preprocess data
    hydat, abund = load_data(hydat_path, abund_path)
    X, C_ref, wav = preprocess_data(hydat, abund, cls=0)
    
    # Resample images
    resample_images('/path/to/img.tif', '/path/to/img2.tif', outdir)
    
    # Multi-objective Optimization with Early Stopping mechanism
    early_stopping = MOEarlyStopping(patience=500, min_delta=1e-4)
    study = optuna.create_study(directions=["maximize", "minimize", "maximize"], sampler=optuna.samplers.NSGAIIISampler())
    study.optimize(lambda trial: objective(trial, X, C_ref), n_trials=2500, callbacks=[early_stopping], n_jobs=4)
    
    # Train and predict
    predicted, predictions = train_and_predict(X, C_ref, global_top_indices, global_feature_mappings, abund.data, cls=0)
    
    # Visualize results
    ref = abund.data[:, :, 0]
    plot_abundance_maps(predicted, ref)
    plot_histograms(predictions, C_ref)
    
    # Calculate metrics
    ssim_pred = calculate_ssim(predicted, ref)
    print(f"Structural Similarity Index (SSIM) for Predicted: {ssim_pred}")
    
    # Calculate means
    pred_float = np.array(predicted, dtype=np.float32)
    ref_float = np.array(ref, dtype=np.float32)
    pred_mean = mean_without_zeros(pred_float)
    ref_mean = mean_without_zeros(ref_float)
    print(f"Reference Mean: {ref_mean}, Predicted Mean: {pred_mean}")
    
    # Save results
    save_results(results_path, joblib.load("best_model.pkl"), study.best_trials, global_top_indices, global_feature_mappings, predictions)
    
    # Load and verify results
    best_trial_info, loaded_top_indices, loaded_feature_mappings = load_results(results_path)
    print("Loaded best trial info:", best_trial_info)
    print("Loaded top indices:", loaded_top_indices)
    print("Loaded feature mappings:", loaded_feature_mappings)
    
    # Print selected wavelengths
    WLnum = []
    for values in global_feature_mappings.values():
        WLnum.extend(values)
    WL = [wav[i] for i in WLnum]
    print("All selected wavelength numbers:", WLnum)
    print("All selected wavelengths:", WL)

if __name__ == "__main__":
    main()