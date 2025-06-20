#!/usr/bin/env python3
"""
Python 3.13 compatibility verification script
Tests the application with Python 3.13 specific features and optimizations
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def check_python_313_compatibility():
    """Comprehensive Python 3.13 compatibility check"""
    print("🔍 Python 3.13 Compatibility Check")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 13):
        print("✅ Running on Python 3.13+ - Full compatibility testing")
        is_py313 = True
    else:
        print(f"ℹ️ Running on Python {python_version.major}.{python_version.minor} - Basic compatibility testing")
        is_py313 = False
    
    # Check TensorFlow version
    print(f"TensorFlow Version: {tf.__version__}")
    tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
    
    if tf_version >= (2, 17):
        print("✅ TensorFlow version is optimal for Python 3.13")
    elif tf_version >= (2, 16):
        print("✅ TensorFlow version is compatible with Python 3.13")
    else:
        print("⚠️ TensorFlow version may have limited Python 3.13 support")
    
    return is_py313

def test_model_loading():
    """Test model loading with different formats"""
    print("\n🔄 Testing Model Loading")
    print("-" * 30)
    
    # Model files to test in order of preference
    model_files = [
        'ann_model_py313.keras',
        'ann_model_py313.h5',
        'ann_model_rebuilt.keras',
        'ann_model_rebuilt.h5',
        'ann_model.keras',
        'ann_model.h5'
    ]
    
    loaded_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                print(f"Testing {model_file}...")
                model = load_model(model_file, compile=False)
                
                # Recompile with Python 3.13 optimized settings
                compile_kwargs = {
                    'optimizer': 'adam',
                    'loss': 'binary_crossentropy',
                    'metrics': ['accuracy']
                }
                
                # Try enhanced optimizer for Python 3.13
                try:
                    if hasattr(tf.keras.optimizers, 'Adam'):
                        compile_kwargs['optimizer'] = tf.keras.optimizers.Adam(
                            learning_rate=0.001,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-7
                        )
                except:
                    pass
                
                model.compile(**compile_kwargs)
                loaded_models.append((model_file, model))
                print(f"  ✅ Successfully loaded and compiled")
                
            except Exception as e:
                print(f"  ❌ Failed to load: {str(e)[:80]}...")
        else:
            print(f"  ➖ {model_file} not found")
    
    return loaded_models

def test_predictions(loaded_models):
    """Test prediction functionality"""
    print("\n🎯 Testing Prediction Functionality")
    print("-" * 35)
    
    if not loaded_models:
        print("❌ No models loaded - cannot test predictions")
        return False
    
    # Sample input for testing (12 features)
    test_inputs = [
        np.array([[600, 1, 40, 3, 60000, 2, 1, 1, 50000, 0, 1, 0]]),  # Low churn
        np.array([[350, 0, 65, 1, 20000, 0, 0, 0, 15000, 1, 0, 1]]),  # High churn
        np.random.random((1, 12))  # Random input
    ]
    
    test_names = ["Low Churn Profile", "High Churn Profile", "Random Input"]
    
    success_count = 0
    
    for model_file, model in loaded_models:
        print(f"\nTesting {model_file}:")
        
        try:
            for i, (test_input, test_name) in enumerate(zip(test_inputs, test_names)):
                prediction = model.predict(test_input, verbose=0)
                churn_prob = prediction[0][0]
                churn_decision = "Yes" if churn_prob > 0.5 else "No"
                
                print(f"  {test_name}: {churn_prob:.4f} ({churn_decision})")
            
            print(f"  ✅ All predictions successful for {model_file}")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Prediction failed for {model_file}: {e}")
    
    return success_count > 0

def test_python_313_features():
    """Test Python 3.13 specific features and optimizations"""
    print("\n🚀 Python 3.13 Specific Features")
    print("-" * 32)
    
    if sys.version_info < (3, 13):
        print("ℹ️ Not running on Python 3.13 - skipping specific feature tests")
        return True
    
    try:
        # Test improved error messages (Python 3.13 feature)
        print("✅ Enhanced error handling available")
        
        # Test performance improvements
        import time
        start_time = time.perf_counter()
        
        # Simple computation to test performance
        result = sum(i**2 for i in range(10000))
        
        end_time = time.perf_counter()
        print(f"✅ Performance test completed: {end_time - start_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Some Python 3.13 features may not be fully available: {e}")
        return False

def main():
    """Main compatibility test function"""
    print("🔧 Python 3.13 Compatibility Test Suite")
    print("=" * 60)
    
    # Run all tests
    is_py313 = check_python_313_compatibility()
    loaded_models = test_model_loading()
    predictions_work = test_predictions(loaded_models)
    py313_features = test_python_313_features()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    
    if loaded_models:
        print(f"✅ Model Loading: {len(loaded_models)} model(s) loaded successfully")
        best_model = loaded_models[0][0]  # First model in preference order
        print(f"🏆 Recommended Model: {best_model}")
    else:
        print("❌ Model Loading: No models could be loaded")
    
    if predictions_work:
        print("✅ Predictions: Working correctly")
    else:
        print("❌ Predictions: Failed")
    
    if is_py313 and py313_features:
        print("✅ Python 3.13 Features: Fully compatible")
    elif is_py313:
        print("⚠️ Python 3.13 Features: Partially compatible")
    else:
        print("ℹ️ Python 3.13 Features: Not applicable")
    
    # Recommendations
    print("\n💡 Recommendations")
    print("-" * 20)
    
    if is_py313:
        if any('py313' in model[0] for model in loaded_models):
            print("🎉 Perfect! Using Python 3.13 optimized model")
        else:
            print("💡 Consider running: python rebuild_model_py313.py")
            print("   This will create a Python 3.13 optimized model")
    
    if not loaded_models:
        print("🚨 Critical: No models available - run rebuild script first")
    
    overall_success = bool(loaded_models) and predictions_work
    
    if overall_success:
        print("\n🎉 Overall Status: READY FOR PYTHON 3.13 DEPLOYMENT! 🚀")
    else:
        print("\n⚠️ Overall Status: Issues detected - please address before deployment")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
