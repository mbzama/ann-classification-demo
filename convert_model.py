#!/usr/bin/env python3
"""
Script to convert the HDF5 model to native Keras format for better compatibility
Run this script to create a .keras version of your model that's more compatible 
with different TensorFlow versions.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def convert_model():
    """Convert HDF5 model to native Keras format"""
    try:
        print(f"TensorFlow version: {tf.__version__}")
        print("Loading HDF5 model...")
        
        # Load the existing model
        model = load_model('ann_model.h5', compile=False)
        
        # Recompile with standard settings
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model loaded successfully!")
        print(f"Model summary:")
        model.summary()
        
        # Save in native Keras format
        print("\nSaving model in native Keras format...")
        model.save('ann_model.keras')
        
        print("✅ Model successfully converted to 'ann_model.keras'")
        print("\nTo use the new format, update your streamlit_app.py:")
        print("  Change: model = load_model('ann_model.h5')")
        print("  To:     model = load_model('ann_model.keras')")
        
        # Verify the converted model works
        print("\nTesting converted model...")
        test_model = load_model('ann_model.keras')
        print("✅ Converted model loads successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure 'ann_model.h5' exists in the current directory")
        print("2. Check TensorFlow installation: pip install tensorflow>=2.15.0")
        print("3. Try running in the same environment where the model was trained")
        return False

if __name__ == "__main__":
    print("🔄 Converting ANN model to native Keras format...")
    print("=" * 50)
    
    if convert_model():
        print("\n🎉 Conversion completed successfully!")
        print("\nNext steps:")
        print("1. Update your Streamlit app to use 'ann_model.keras'")
        print("2. Add 'ann_model.keras' to your git repository")
        print("3. You can remove 'ann_model.h5' if desired")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")
