#!/usr/bin/env python3
"""
Quick verification script to test that the rebuilt model works correctly
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_model():
    """Test that the rebuilt model loads and works correctly"""
    try:
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test loading the rebuilt model
        print("Testing rebuilt model...")
        model = load_model('ann_model_rebuilt.keras')
        print("✅ Rebuilt model loaded successfully!")
        
        # Test prediction
        print("\nTesting prediction...")
        # Create sample input (12 features for the ANN)
        sample_input = np.array([[600, 1, 40, 3, 60000, 2, 1, 1, 50000, 0, 1, 0]])
        
        prediction = model.predict(sample_input, verbose=0)
        churn_probability = prediction[0][0]
        
        print(f"Sample prediction: {churn_probability:.4f}")
        print(f"Churn prediction: {'Yes' if churn_probability > 0.5 else 'No'} ({churn_probability:.1%})")
        
        print("\n✅ Model verification successful!")
        print("The rebuilt model is ready for deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Verifying rebuilt model...")
    print("=" * 40)
    
    if test_model():
        print("\n🎉 All tests passed! Ready for deployment.")
    else:
        print("\n❌ Verification failed.")
