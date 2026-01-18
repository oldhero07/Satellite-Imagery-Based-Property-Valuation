import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Constants
IMAGE_DIR = "satellite_images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Proper training

# Load Data
train_df = pd.read_excel('train.xlsx')
test_df = pd.read_excel('test.xlsx')

numerical_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

train_df[numerical_cols] = train_df[numerical_cols].fillna(0)
test_df[numerical_cols] = test_df[numerical_cols].fillna(0)

scaler = StandardScaler()
X_train_num_all = scaler.fit_transform(train_df[numerical_cols])
y_train_all = train_df['price'].values
X_test_num_all = scaler.transform(test_df[numerical_cols])

# --- Helper to load images ---
def load_batch_images(ids, img_dir=IMAGE_DIR, dim=IMG_SIZE):
    images = []
    for ID in ids:
        img_path = os.path.join(img_dir, f"{ID}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, dim)
                img = img / 255.0
            else:
                img = np.zeros((*dim, 3))
        else:
            img = np.zeros((*dim, 3))
        images.append(img)
    return np.array(images, dtype=np.float32)

# --- Train on Subset (for Stability/Speed verification) ---
SUBSET_SIZE = 2000  # Use more data for better training
print(f"Training on subset of {SUBSET_SIZE} samples...")
train_ids_subset = train_df['id'].values[:SUBSET_SIZE]
X_num_subset = X_train_num_all[:SUBSET_SIZE].astype(np.float32)
y_subset = y_train_all[:SUBSET_SIZE].astype(np.float32)

# Normalize target values to prevent scale issues
y_mean = np.mean(y_subset)
y_std = np.std(y_subset)
y_subset_normalized = (y_subset - y_mean) / y_std

X_img_subset = load_batch_images(train_ids_subset)

# Build Model
def create_multimodal_model():
    input_num = layers.Input(shape=(X_train_num_all.shape[1],))
    x_num = layers.Dense(64, activation='relu')(input_num) # Reduced size
    
    input_img = layers.Input(shape=(224, 224, 3))
    x_img = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)
    x_img = layers.Flatten()(x_img)
    x_img = layers.Dense(32, activation='relu')(x_img)

    combined = layers.concatenate([x_num, x_img])
    output = layers.Dense(1, activation='linear')(combined)

    model = models.Model(inputs=[input_num, input_img], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = create_multimodal_model()
model.fit([X_num_subset, X_img_subset], y_subset_normalized, epochs=EPOCHS, batch_size=16, verbose=1)
model.save('multimodal_model.h5')
print("Model saved.")

# --- Predict on All Test Data (Batched Manually) ---
print("Predicting on test set...")
predictions = []
ids = test_df['id'].values
num_batches = int(np.ceil(len(ids) / BATCH_SIZE))

for i in tqdm(range(num_batches)):
    batch_ids = ids[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
    batch_num = X_test_num_all[i*BATCH_SIZE : (i+1)*BATCH_SIZE].astype(np.float32)
    batch_img = load_batch_images(batch_ids)
    
    preds = model.predict_on_batch([batch_num, batch_img])
    # Denormalize predictions back to original scale
    preds_denormalized = (preds.flatten() * y_std) + y_mean
    predictions.extend(preds_denormalized)

submission_df = pd.DataFrame({'id': ids, 'predicted_price': predictions})
submission_df.to_csv('24119021_final.csv', index=False)
print("Saved 24119021_final.csv")
