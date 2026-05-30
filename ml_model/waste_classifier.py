"""
Waste Classification Model Wrapper
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import pickle
from typing import Dict, Optional, Any


class WasteClassifier:
    # Singleton class to load and use the waste classification model.
    
    _instance = None
    _model: Optional[tf.keras.Model] = None
    _label_map: Optional[Dict] = None
    _idx_to_label: Dict[int, str] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        """Load the trained model and label map."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'ml_model', 'models', 'waste_classifier.h5')
        label_path = os.path.join(base_dir, 'ml_model', 'models', 'label_map.pkl')
        
        print(f"Looking for model at: {model_path}")
        
        try:
            if os.path.exists(model_path):
                self._model = tf.keras.models.load_model(model_path)
                print(" Model loaded successfully!")
            else:
                print(f" Model not found at {model_path}")
                print("   Please train the model first:")
                print("   cd ml_model && python train_model.py")
                self._model = None
                return
            
            if os.path.exists(label_path):
                with open(label_path, 'rb') as f:
                    self._label_map = pickle.load(f)
                
                # Check if label_map is not None before processing
                if self._label_map is not None:
                    # Convert to regular dict and ensure keys are integers
                    self._idx_to_label = {int(v): str(k) for k, v in self._label_map.items()}
                    print(f" Label map loaded: {list(self._label_map.keys())}")
                else:
                    print(" Label map file contains no data, using defaults")
                    self._use_default_labels()
            else:
                print(" Label map not found, using defaults")
                self._use_default_labels()
                
        except Exception as e:
            print(f" Error loading model: {e}")
            self._model = None
            self._use_default_labels()
    
    def _use_default_labels(self):
        """Set default label mapping when file is not available."""
        self._idx_to_label = {
            0: 'cardboard', 
            1: 'glass', 
            2: 'metal', 
            3: 'paper', 
            4: 'plastic', 
            5: 'trash'
        }
        print(f" Using default labels: {list(self._idx_to_label.values())}")
    
    def classify_image(self, image_array):
        """Classify a single image array."""
        if self._model is None:
            return {'error': 'Model not loaded', 'label': 'unknown', 'confidence': 0.0}
        
        try:
            # Preprocess image
            resized = cv2.resize(image_array, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb / 255.0
            input_tensor = np.expand_dims(normalized, axis=0)
            
            # Predict
            predictions = self._model.predict(input_tensor, verbose=0)
            class_id = int(np.argmax(predictions[0]))  # Convert to int explicitly
            confidence = float(predictions[0][class_id])
            
            # Safe label lookup - check if _idx_to_label has the key
            if self._idx_to_label and class_id in self._idx_to_label:
                label = self._idx_to_label[class_id]
            else:
                label = f'class_{class_id}'
            
            return {
                'class_id': class_id,
                'label': label,
                'confidence': confidence,
                'all_predictions': predictions[0].tolist()
            }
            
        except Exception as e:
            print(f" Classification error: {e}")
            return {'error': str(e), 'label': 'unknown', 'confidence': 0.0}
    
    def is_ready(self):
        """Check if model is ready."""
        return self._model is not None


if __name__ == "__main__":
    print("Testing WasteClassifier...")
    classifier = WasteClassifier()
    
    if classifier.is_ready():
        print(" Classifier is ready to use!")
        print(f"   Labels: {list(classifier._idx_to_label.values())}")
    else:
        print(" Classifier not ready. Please train the model first.")