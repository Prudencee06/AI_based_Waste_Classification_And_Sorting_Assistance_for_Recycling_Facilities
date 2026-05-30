print("Testing imports...")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
    
    from keras import models
    print("✅ tensorflow.keras.layers, models")
    
    from keras.preprocessing.image import load_img, img_to_array
    print("✅ Image utilities")
    
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    print("✅ Callbacks")
    
    print("\n🎉 All imports working! You can train the model.")
    
except Exception as e:
    print(f"❌ Error: {e}")