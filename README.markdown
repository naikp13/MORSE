# MORSE: Model Optimization Tools for Remote Sensing Data Analysis

MORSE (Model Optimization tools for Remote Sensing Data Analysis) is a Python-based framework for optimizing machine learning models for multi-modal remote sensing data. It leverages "Optuna" for hyperparameter optimization, SHAP for feature importance, and various ML models for EO applications. Additionally, it supports feature selection using Optuna's TPESampler.

## Features

- Data preprocessing for remote sensing data and stack objects.
- Multi-objective optimization using Optuna's NSGA-III sampler for ML models.
- Band ratio-based feature selection using Optuna's TPESampler.
- Visualization of predicted maps and spectral plots.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/MORSE.git
   cd MORSE
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. For GPU acceleration, install `cupy` with the appropriate CUDA version (optional).

## Usage

1. Place your hyperspectral data (e.g., `.hdr`, or `.stack.joblib` files) in the appropriate directory.
2. Run the main script for regression-based abundance prediction:

   ```bash
   python src/main.py
   ```
3. Run the feature selection script for band ratio optimization:

   ```bash
   python src/main_feature_selection.py
   ```
4. Results (models, predictions, visualizations, and band ratios) will be saved to the specified output directory.

## Directory Structure

- `src/`: Source code for preprocessing, optimization, training, visualization, and feature selection.
- `data/`: Store your  data here .
- `results/`: Output directory for models, predictions, and visualizations.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.