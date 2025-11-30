# MORSE: Model Optimization Tools for Remote Sensing Data Analysis

MORSE (Model Optimization tools for Remote Sensing Data Analysis) is a Python-based framework for optimizing machine learning models for multi-modal remote sensing data. It leverages "Optuna" for hyperparameter optimization, SHAP for feature importance and explainability, and various ML models for EO applications. Additionally, it supports feature selection using TPESampler.

A few appplications of this framework at showcased in the following publications -
1. https://doi.org/10.1109/whispers65427.2024.10876480
2. https://doi.org/10.13140/RG.2.2.25599.73123

<p align="center">
  <img src="https://raw.githubusercontent.com/naikp13/MORSE/main/imgs/morse_fig1.png" alt="pareto-front" width="400"/>
  <img src="https://raw.githubusercontent.com/naikp13/MORSE/main/imgs/morse_fig2.png" alt="parento optimal solution" width="400"/>
</p>

## Features

- Data preprocessing for remote sensing data and stack objects.
- Multi-objective optimization using NSGA-III sampler (Genetic Optimization) for ML models.
- Dynamic Feature selection using TPESampler (Bayesian Optimization).
- Visualization of predicted maps and spectral plots.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/naikp13/MORSE.git
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
- 
## Cite this work

Please cite this article in case this method was helpful for your research or used for your work,

```Citation
Naik, P., Chakraborty, R., Thiele, S., Kirsch, M., & Gloaguen, R. (2024). Multi-Objective Optimization Based Hyperspectral Feature Engineering for Spectral Abundance Mapping. In 2024 14th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS) (pp. 1–5). 2024 14th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS). IEEE. https://doi.org/10.1109/whispers65427.2024.10876480
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions or bug reports.
