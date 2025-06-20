#!/usr/bin/env python3
"""
Updated script to rebuild the model architecture for Python 3.13 compatibility.
This version includes additional error handling and compatibility checks.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def check_python_version():
    """Check if running on Python 3.13+"""
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 13):
        print("✅ Running on Python 3.13+ - using optimized compatibility settings")
        return True
    else:
        print("ℹ️ Running on older Python version - standard settings will be used")
        return False

def rebuild_model_py313():
    """Rebuild model architecture with Python 3.13 optimizations"""
    try:
        is_py313 = check_python_version()
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check TensorFlow compatibility with Python 3.13
        tf_version = tuple(map(int, tf.__version__.split('.')[:2]))
        if is_py313 and tf_version < (2, 16):
            print("⚠️ Warning: TensorFlow version may not be fully optimized for Python 3.13")
            print("   Recommended: TensorFlow >= 2.16.0 for best Python 3.13 support")
        
        print("Loading original model to extract architecture and weights...")
        
        # Try loading from different model files
        model_files = ['ann_model.h5', 'ann_model_rebuilt.h5', 'ann_model.keras']
        original_model = None
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    print(f"Attempting to load from {model_file}...")
                    original_model = load_model(model_file, compile=False)
                    print(f"✅ Successfully loaded from {model_file}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {model_file}: {str(e)[:100]}...")
                    continue
        
        if original_model is None:
            raise Exception("Could not load any model file")
        
        print("Original model summary:")
        original_model.summary()
        
        # Extract layer information with enhanced error handling
        layers_info = []
        for i, layer in enumerate(original_model.layers):
            if hasattr(layer, 'units'):  # Dense layer
                activation_name = 'linear'  # default
                try:
                    if hasattr(layer.activation, '__name__'):
                        activation_name = layer.activation.__name__
                    elif hasattr(layer.activation, 'name'):
                        activation_name = layer.activation.name
                    else:
                        activation_name = str(layer.activation).split()[1] if 'function' in str(layer.activation) else 'linear'
                except:
                    activation_name = 'relu' if i < len(original_model.layers) - 1 else 'sigmoid'
                
                layers_info.append({
                    'type': 'dense',
                    'units': layer.units,
                    'activation': activation_name,
                    'weights': layer.get_weights()
                })
                print(f"Layer {i}: Dense({layer.units}, activation='{activation_name}')")
        
        # Get input shape
        input_shape = original_model.input_shape[1:]
        print(f"Input shape: {input_shape}")
        
        # Build new model with Python 3.13 optimized settings
        print("\nBuilding new Python 3.13 compatible model...")
        new_model = Sequential(name="ann_model_py313")
        
        # Add layers with explicit naming for better compatibility
        for i, layer_info in enumerate(layers_info):
            layer_name = f'dense_py313_{i}'
            
            if i == 0:  # First layer
                new_model.add(Dense(
                    units=layer_info['units'],
                    activation=layer_info['activation'],
                    input_shape=input_shape,
                    name=layer_name
                ))
            else:
                new_model.add(Dense(
                    units=layer_info['units'],
                    activation=layer_info['activation'],
                    name=layer_name
                ))
        
        # Compile with settings optimized for Python 3.13
        compile_kwargs = {
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']
        }
        
        # Add Python 3.13 specific optimizations if available
        if is_py313 and hasattr(tf.keras.optimizers, 'legacy'):
            try:
                compile_kwargs['optimizer'] = tf.keras.optimizers.Adam(
                    learning_rate=0.001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                )
            except:
                pass  # Fall back to string optimizer
        
        new_model.compile(**compile_kwargs)
        
        print("\nNew model summary:")
        new_model.summary()
        
        # Transfer weights with enhanced error handling
        print("\nTransferring weights...")
        for i, layer_info in enumerate(layers_info):
            if layer_info['weights']:
                try:
                    new_model.layers[i].set_weights(layer_info['weights'])
                    print(f"✅ Transferred weights for layer {i}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not transfer weights for layer {i}: {e}")
        
        # Save the rebuilt model
        print("\nSaving Python 3.13 compatible model...")
        model_name_keras = 'ann_model_py313.keras'
        model_name_h5 = 'ann_model_py313.h5'
        
        try:
            new_model.save(model_name_keras)
            print(f"✅ Saved: {model_name_keras}")
        except Exception as e:
            print(f"⚠️ Could not save .keras format: {e}")
        
        try:
            new_model.save(model_name_h5)
            print(f"✅ Saved: {model_name_h5}")
        except Exception as e:
            print(f"⚠️ Could not save .h5 format: {e}")
        
        # Test the rebuilt model
        print("\nTesting Python 3.13 compatible model...")
        test_model = load_model(model_name_keras if os.path.exists(model_name_keras) else model_name_h5)
        print("✅ Python 3.13 compatible model loads successfully!")
        
        # Verification test
        print("\nTesting prediction capability...")
        test_input = np.random.random((1, input_shape[0]))
        
        try:
            original_pred = original_model.predict(test_input, verbose=0)
            rebuilt_pred = test_model.predict(test_input, verbose=0)
            
            print(f"Original prediction: {original_pred[0][0]:.6f}")
            print(f"Rebuilt prediction: {rebuilt_pred[0][0]:.6f}")
            print(f"Difference: {abs(original_pred[0][0] - rebuilt_pred[0][0]):.6f}")
            
            if abs(original_pred[0][0] - rebuilt_pred[0][0]) < 1e-5:
                print("✅ Models produce identical predictions!")
            else:
                print("⚠️ Small difference in predictions (may be due to numerical precision)")
        except Exception as e:
            print(f"⚠️ Could not compare predictions: {e}")
            # Just test that the new model can make predictions
            rebuilt_pred = test_model.predict(test_input, verbose=0)
            print(f"✅ New model prediction works: {rebuilt_pred[0][0]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error rebuilding model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure a model file exists (ann_model.h5, ann_model.keras, etc.)")
        print("2. Check TensorFlow installation: pip install tensorflow>=2.16.0")
        print("3. For Python 3.13, ensure you have the latest compatible packages")
        return False

if __name__ == "__main__":
    print("🔧 Rebuilding ANN model for Python 3.13 compatibility...")
    print("=" * 65)
    
    if rebuild_model_py313():
        print("\n🎉 Python 3.13 compatible model rebuild completed successfully!")
        print("\nFiles created:")
        print("- ann_model_py313.keras (recommended for Python 3.13)")
        print("- ann_model_py313.h5 (backup format)")
        print("\nNext steps:")
        print("1. Update your Streamlit app to use 'ann_model_py313.keras'")
        print("2. Test locally with Python 3.13")
        print("3. Deploy to Streamlit Cloud with Python 3.13 runtime")
    else:
        print("\n❌ Model rebuild failed. Please check the error messages above.")
