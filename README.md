# Satellite Imagery Based Property Valuation

## Overview
This project predicts property prices based on satellite imagery and other features.

## Components
- `data_fetcher.py`: Fetches satellite images.
- `preprocessing.ipynb`: Preprocesses data for training.
- `model_training.ipynb`: Trains the valuation model.
- `enrollno_final.csv`: Prediction output.
- `enrollno_report.pdf`: Project report.

## Setup
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow opencv-python requests tqdm openpyxl xlrd
   ```
2. **Data**: Ensure `train.xlsx` and `test.xlsx` are in the project root. (Already downloaded).

## Execution Steps
1. **Fetch Satellite Images**:
   ```bash
   python data_fetcher.py
   ```
   *Note: This downloads images to `satellite_images/`. It may take time.*

2. **Train Model & Generate Predictions**:
   Open `model_training.ipynb` and run all cells.
   *Alternatively, for a quick run:*
   ```bash
   python train_model.py
   ```
   This generates `enrollno_final.csv` and saves the model to `multimodal_model.h5`.

3. **Generate Report**:
   Open `report_generator.ipynb`, run all cells to visualize findings and Grad-CAM.
   **To create PDFs**:
   - Open `report_generator.html` in your browser and "Print to PDF".
   - Save as `24119021_report.pdf`.

## Deliverables
- `24119021_final.csv`: Predictions.
- `24119021_report.pdf`: Analysis.
- Codebase.
