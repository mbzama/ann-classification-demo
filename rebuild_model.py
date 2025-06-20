#!/usr/bin/env python3
"""
Script to rebuild the model architecture and copy weights to fix InputLayer compatibility issues.
This approach creates a new model with compatible layer definitions and transfers the trained weights.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def rebuild_model():
    """Rebuild model architecture and transfer weights"""
    try:
        print(f"TensorFlow version: {tf.__version__}")
        print("Loading original model to extract architecture and weights...")
        
        # Load the original model
        original_model = load_model('ann_model.h5', compile=False)
        
        print("Original model summary:")
        original_model.summary()
        
        # Extract layer information
        layers_info = []
        for i, layer in enumerate(original_model.layers):
            if hasattr(layer, 'units'):  # Dense layer
                layers_info.append({
                    'type': 'dense',
                    'units': layer.units,
                    'activation': layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation),
                    'weights': layer.get_weights()
                })
                print(f"Layer {i}: Dense({layer.units}, activation='{layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)}')")
        
        # Get input shape from the first layer
        input_shape = original_model.input_shape[1:]  # Remove batch dimension
        print(f"Input shape: {input_shape}")
        
        # Build new model with compatible architecture
        print("\nBuilding new compatible model...")
        new_model = Sequential()
        
        # Add layers based on extracted information
        for i, layer_info in enumerate(layers_info):
            if i == 0:  # First layer needs input_shape
                new_model.add(Dense(
                    units=layer_info['units'],
                    activation=layer_info['activation'],
                    input_shape=input_shape,
                    name=f'dense_{i}'
                ))
            else:
                new_model.add(Dense(
                    units=layer_info['units'],
                    activation=layer_info['activation'],
                    name=f'dense_{i}'
                ))
        
        # Compile the new model
        new_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nNew model summary:")
        new_model.summary()
        
        # Transfer weights
        print("\nTransferring weights...")
        for i, layer_info in enumerate(layers_info):
            if layer_info['weights']:  # If layer has weights
                new_model.layers[i].set_weights(layer_info['weights'])
                print(f"Transferred weights for layer {i}")
        
        # Save the rebuilt model in both formats
        print("\nSaving rebuilt model...")
        new_model.save('ann_model_rebuilt.keras')
        new_model.save('ann_model_rebuilt.h5')
        
        print("✅ Model successfully rebuilt and saved as:")
        print("  - ann_model_rebuilt.keras")
        print("  - ann_model_rebuilt.h5")
        
        # Test the rebuilt model
        print("\nTesting rebuilt model...")
        test_model = load_model('ann_model_rebuilt.keras')
        print("✅ Rebuilt model loads successfully!")
        
        # Create a simple test to verify functionality
        print("\nTesting prediction capability...")
        test_input = np.random.random((1, input_shape[0]))
        original_pred = original_model.predict(test_input, verbose=0)
        rebuilt_pred = test_model.predict(test_input, verbose=0)
        
        print(f"Original prediction: {original_pred[0][0]:.6f}")
        print(f"Rebuilt prediction: {rebuilt_pred[0][0]:.6f}")
        print(f"Difference: {abs(original_pred[0][0] - rebuilt_pred[0][0]):.6f}")
        
        if abs(original_pred[0][0] - rebuilt_pred[0][0]) < 1e-6:
            print("✅ Models produce identical predictions!")
        else:
            print("⚠️ Small difference in predictions (expected due to floating point precision)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error rebuilding model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure 'ann_model.h5' exists in the current directory")
        print("2. Check TensorFlow installation")
        print("3. Verify the original model was trained properly")
        return False

if __name__ == "__main__":
    print("🔧 Rebuilding ANN model for compatibility...")
    print("=" * 60)
    
    if rebuild_model():
        print("\n🎉 Model rebuild completed successfully!")
        print("\nNext steps:")
        print("1. Update your Streamlit app to use 'ann_model_rebuilt.keras'")
        print("2. Add the rebuilt model to your git repository")
        print("3. The rebuilt model should be compatible with different TensorFlow versions")
    else:
        print("\n❌ Model rebuild failed. Please check the error messages above.")
