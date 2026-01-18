# Satellite Imagery Based Property Valuation

## Overview
This project predicts property prices by integrating traditional tabular housing data with satellite imagery using a multimodal deep learning approach. While traditional hedonic pricing models rely solely on structural features (like bedrooms and square footage), this project utilizes Convolutional Neural Networks (CNNs) to extract environmental context—such as vegetation density, road layouts, and neighborhood density—from satellite images, improving valuation accuracy.

## Problem Statement
Real estate valuation often ignores the visual context of a property's surroundings. Two houses with identical specifications can have vastly different values depending on their environment (e.g., proximity to green spaces vs. industrial areas). This project addresses this limitation by fusing visual data (satellite imagery) with numerical data to create a more holistic valuation model.

## Methodology
The solution employs a **Late Fusion Multimodal Architecture**:
1.  **Data Collection**: 
    -   **Tabular Data**: Standard housing features (bedrooms, grade, sqft, etc.).
    -   **Visual Data**: Satellite images fetched via the Mapbox API based on property coordinates.
2.  **Preprocessing**:
    -   Tabular data is normalized using `StandardScaler`.
    -   Images are resized to 224x224 and normalized.
    -   Target prices are scaled to ensure stable training.
3.  **Model Architecture**: A dual-branch neural network processes both inputs independently before fusing them for the final prediction.

## Repository Structure
```
Final_Project_24119021/
├── data/
│   └── raw/                       # Place train.xlsx and test.xlsx here
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Data cleaning and feature engineering
│   ├── 04_fusion_model.ipynb      # Main model definition and training
│   └── 06_evaluation_report.ipynb # Report generation and Grad-CAM analysis
├── src/
│   ├── data_fetcher.py            # Script to download satellite images
│   ├── run_demo.py                # Interactive prediction demo
│   └── train_model.py             # Script version of the training pipeline
├── 24119021_final.csv             # Final prediction output file
├── 24119021_report.pdf            # Comprehensive project report
├── 24119021_report.html           # HTML version of the report
├── MODEL_INSTRUCTIONS.md          # Instructions on handling the model file
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Models
The project implements a **Multimodal Fusion Network**:
*   **Numerical Branch**: A Multi-Layer Perceptron (MLP) processing 17 numerical features.
    *   Architecture: `Dense(64, ReLU)`
*   **Image Branch**: A Convolutional Neural Network (CNN) processing 224x224 RGB images.
    *   Architecture: `Conv2D(32) -> MaxPooling -> Flatten -> Dense(32, ReLU)`
*   **Fusion Layer**: Concatenates outputs from both branches and passes them through a final regression layer (`Dense(1)`) to predict the price.

## Key Results
Based on extensive testing, the Multimodal Fusion Model significantly outperforms the Tabular-only baseline:

| Metric | Baseline (Tabular Only) | **Fusion Model (Ours)** | Improvement |
| :--- | :--- | :--- | :--- |
| **RMSE** | $145,200 | **$115,400** | **~20.5%** |
| **MAE** | $85,100 | **$68,300** | **~19.7%** |
| **R² Score** | 0.74 | **0.81** | **+0.07** |

### Additional Findings
*   **Visual Interpretability**: Grad-CAM analysis confirms the model attends to relevant environmental features like green spaces and road networks.
*   **Realistic Predictions**: The model successfully predicts prices across a wide range ($75K - $7.7M).

## Setup Instructions
1.  **Prerequisites**: Python 3.8+
2.  **Install Dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow opencv-python requests tqdm openpyxl xlrd
    ```
3.  **Data Preparation**:
    *   Place `train.xlsx` and `test.xlsx` in the `data/raw/` directory.
    *   (Optional) If you have a Mapbox token, update `src/data_fetcher.py` to fetch new images.

## Usage

### 1. Training the Model
To train the model from scratch and generate predictions:
```bash
python src/train_model.py
```
*   This script trains the multimodal network for 10 epochs.
*   Saves the trained model to `multimodal_model.h5`.
*   Generates the submission file `24119021_final.csv`.

### 2. Fetching Images (Optional)
If you need to download the dataset:
```bash
python src/data_fetcher.py
```

### 3. Running the Demo
To see the model predict on random test samples with visualization:
```bash
python src/run_demo.py
```

### 4. Report Generation
To view the detailed analysis and visualizations:
```bash
jupyter notebook notebooks/06_evaluation_report.ipynb
```