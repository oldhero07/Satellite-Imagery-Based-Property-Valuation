# Model File Instructions

## Trained Model Location
The trained model `multimodal_model.h5` (~2MB) is excluded from Git due to file size limits.

## How to Generate the Model
1. Run the training script:
   ```bash
   python train_model.py
   ```
   This will create `multimodal_model.h5` in the project directory.

2. Alternatively, run the Jupyter notebook:
   ```bash
   jupyter notebook model_training.ipynb
   ```
   Execute all cells to train and save the model.

## Model Architecture
- **Input 1**: Numerical features (17 dimensions)
- **Input 2**: Satellite images (224x224x3)
- **CNN Branch**: Conv2D → MaxPooling → Flatten → Dense(32)
- **Numerical Branch**: Dense(64)
- **Fusion**: Concatenate → Dense(1) output
- **Training**: 10 epochs, normalized targets, proper scaling

## Expected Performance
- **Training Loss**: ~0.005 (normalized scale)
- **Prediction Range**: $75K - $7.7M (realistic property prices)
- **Mean Prediction**: ~$493K (close to actual ~$537K)

The model file will be automatically generated when you run the training code.