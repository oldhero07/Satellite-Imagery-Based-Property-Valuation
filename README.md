# Satellite Imagery Based Property Valuation

## Overview
This project predicts property prices based on satellite imagery and other features using a multimodal deep learning approach.

## Components
- `data_fetcher.py`: Fetches satellite images using Mapbox API
- `preprocessing.ipynb`: Data cleaning and feature engineering
- `model_training.ipynb`: Interactive multimodal model training
- `train_model.py`: Automated training script with proper scaling
- `24119021_final.csv`: Final prediction output
- `24119021_report.pdf`: Complete project analysis report

## Setup
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow opencv-python requests tqdm openpyxl xlrd
   ```
2. **Data**: Ensure `train.xlsx` and `test.xlsx` are in the project root.

## Execution Steps

### 1. Fetch Satellite Images:
```bash
python data_fetcher.py
```
*Note: This downloads ~2000 images to `satellite_images/` folder. May take 10-15 minutes.*

### 2. Train Model & Generate Predictions:

**Option A - Quick Training:**
```bash
python train_model.py
```
This generates `24119021_final.csv` and saves the model to `multimodal_model.h5`.

**Option B - Interactive Training:**
Open `model_training.ipynb` and run all cells for detailed analysis.

### 3. Generate Report:
Open `24119021_report_generator.ipynb` and run all cells to create visualizations and Grad-CAM analysis.

## Model Architecture
- **Multimodal CNN**: Combines satellite imagery (CNN) with tabular features
- **Image Processing**: 224x224 RGB satellite images via Mapbox API
- **Features**: 17 numerical features (bedrooms, bathrooms, sqft_living, etc.)
- **Target Normalization**: Proper price scaling for realistic predictions
- **Explainability**: Grad-CAM visualization for model interpretability

## Key Features
- ✅ **Satellite Image Integration**: Automated fetching via Mapbox API
- ✅ **Multimodal Architecture**: CNN + Tabular data fusion
- ✅ **Model Explainability**: Grad-CAM heatmaps
- ✅ **Baseline Comparison**: Tabular-only vs Multimodal performance
- ✅ **Proper Scaling**: Realistic price predictions ($75K - $7.7M range)

## Deliverables
- `24119021_final.csv`: Property price predictions
- `24119021_report.pdf`: Complete analysis and findings
- Complete codebase with documentation

## File Size Notes
- `satellite_images/` folder (~200MB) and `multimodal_model.h5` (~2MB) are excluded from Git due to size
- Run `data_fetcher.py` to download satellite images
- Run `train_model.py` to generate the trained model

### 4. Interactive Live Demo:
To see the model in action (predicting on random test samples):
```bash
python run_demo.py
```
*   Displays the satellite image.
*   Shows the real-time price prediction.

## Results
The multimodal model achieves improved prediction accuracy by incorporating visual environmental context from satellite imagery alongside traditional property features.