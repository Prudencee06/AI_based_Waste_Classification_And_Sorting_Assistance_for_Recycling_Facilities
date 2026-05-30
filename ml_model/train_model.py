# Train a waste classification model using local dataset.

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle


def create_model(num_classes):
    # Create a CNN model for waste classification.
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model():
    print("=" * 60)
    print("   WASTE CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    
    # Check if dataset exists
    dataset_path = 'data/data'
    if not os.path.exists(dataset_path):
        print(f"\n Dataset not found at: {dataset_path}")
        print("Please place your dataset in: ml_model/data/data/")
        return None
    
    # Get categories
    categories = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
    categories.sort()
    num_classes = len(categories)
    label_map = {cat: idx for idx, cat in enumerate(categories)}
    
    print(f"\n Found {num_classes} categories:")
    for cat in categories:
        cat_path = os.path.join(dataset_path, cat)
        num_images = len(os.listdir(cat_path))
        print(f"   {cat}: {num_images} images")
    
    # Image parameters
    img_size = (224, 224)
    batch_size = 32
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    print("\n Loading training data...")
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"\n Training samples: {train_generator.samples}")
    print(f" Validation samples: {validation_generator.samples}")
    
    # Create model
    print("\n Building model...")
    model = create_model(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Use H5 format
    checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    
    print("\n Training model for 20 epochs...")
    print("   This will take 20-40 minutes...")
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Save final model
    model.save('models/waste_classifier.h5')
    
    with open('models/label_map.pkl', 'wb') as f:
        pickle.dump(label_map, f)
    
    print("\n" + "=" * 60)
    print("    MODEL SAVED SUCCESSFULLY!")
    print("=" * 60)
    print(f"   Model: models/waste_classifier.h5")
    print(f"   Label map: models/label_map.pkl")
    
    test_loss, test_acc = model.evaluate(validation_generator, verbose=0)
    print(f"\n Validation Accuracy: {test_acc:.2%}")
    
    return model, history


if __name__ == "__main__":
    result = train_model()
    if result:
        print("\n Training complete! Run: python realtime/camera_detector.py")