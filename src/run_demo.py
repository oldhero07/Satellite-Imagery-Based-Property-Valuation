import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Constants
IMAGE_DIR = "satellite_images"
MODEL_PATH = "multimodal_model.h5"

def run_demo():
    print("--- Loading System ---")
    
    # 1. Load Data
    print("Loading test data...")
    test_df = pd.read_excel('test.xlsx')
    train_df = pd.read_excel('train.xlsx') # Needed for fitting scaler
    
    # 2. Prepare Scaler
    print("Initializing feature scaler...")
    numerical_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    train_df[numerical_cols] = train_df[numerical_cols].fillna(0)
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_cols])
    
    # 3. Load Model
    print(f"Loading AI Model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Calculate Price Stats for Denormalization (to show real $ values)
    price_mean = train_df['price'].mean()
    price_std = train_df['price'].std()

    # Filter Test DF to only include IDs we have images for (for smooth demo)
    available_images = set([f.split('.')[0] for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    test_df['id'] = test_df['id'].astype(str)
    test_df = test_df[test_df['id'].isin(available_images)]
    
    if len(test_df) == 0:
        print("Error: No images found in satellite_images/ folder matching test.xlsx IDs.")
        return

    while True:
        print("\n" + "="*40)
        user_input = input(f"Press ENTER to predict on a random property (Pool: {len(test_df)} images) (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
            
        # Pick random sample
        random_row = test_df.sample(1).iloc[0]
        pid = random_row['id']
        print(f"Analyzing Property ID: {pid}")
        
        # Check Image
        img_path = os.path.join(IMAGE_DIR, f"{pid}.jpg")
        if not os.path.exists(img_path):
            print("Satellite image not found locally, skipping...")
            continue
            
        # Prepare inputs
        img = cv2.imread(img_path)
        img_input = cv2.resize(img, (224, 224)) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        # Prepare Features (Use DataFrame to avoid sklearn name warning)
        features_df = pd.DataFrame([random_row[numerical_cols]], columns=numerical_cols)
        features_scaled = scaler.transform(features_df)
        
        # Predict
        print("Running Multimodal Inference...")
        pred_normalized = model.predict([features_scaled, img_input], verbose=0)[0][0]
        
        # Denormalize to get explicit dollar amount
        pred_price = (pred_normalized * price_std) + price_mean
        
        # Display
        print(f"SUCCESS! Predicted Price: ${pred_price:,.2f}")
        print("Showing satellite imagery...")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"ID: {pid}\nPred: ${pred_price:,.2f}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    run_demo()
