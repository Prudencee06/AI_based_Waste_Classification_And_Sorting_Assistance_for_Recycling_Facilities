"""
Data preprocessing for waste classification.
Handles loading, resizing, normalizing, and augmenting images.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


def load_dataset(data_path, img_size=(224, 224)):
    """
    Load images from folder structure.
    
    Args:
        data_path: Path to dataset folder with category subfolders
        img_size: Target size (height, width)
    
    Returns:
        X: Array of images
        y: Array of labels
        label_map: Dictionary mapping category names to numbers
    """
    # Get all categories (subfolders)
    categories = [d for d in os.listdir(data_path) 
                  if os.path.isdir(os.path.join(data_path, d))]
    categories.sort()
    
    label_map = {cat: idx for idx, cat in enumerate(categories)}
    print(f"Categories: {label_map}")
    
    X = []
    y = []
    
    for category in categories:
        category_path = os.path.join(data_path, category)
        print(f"Loading {category} images...")
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, img_size)
            
            # Normalize to [0, 1]
            img = img / 255.0
            
            X.append(img)
            y.append(label_map[category])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"Loaded {len(X)} images")
    print(f"Image shape: {X[0].shape if len(X) > 0 else 'No images'}")
    
    return X, y, label_map


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets.
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: separate validation from remaining
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, 
        random_state=random_state
    )
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, label_map, output_dir='processed_data'):
    """
    Save preprocessed data to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    with open(os.path.join(output_dir, 'label_map.pkl'), 'wb') as f:
        pickle.dump(label_map, f)
    
    print(f"Data saved to {output_dir}/")


def create_data_augmentation():
    """
    Create data augmentation generator.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


if __name__ == "__main__":
    # Test the preprocessing
    data_path = "data/data"  # Path to extracted dataset
    
    if os.path.exists(data_path):
        X, y, label_map = load_dataset(data_path)
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Save processed data
        save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, label_map)
        
        print("\n✅ Preprocessing complete!")
    else:
        print(f"Dataset not found at {data_path}")
        print("Please run train_model.py first to download the dataset.")