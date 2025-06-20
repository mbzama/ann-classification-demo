#!/usr/bin/env python3
"""
Run the Customer Churn Prediction Streamlit App
"""
import subprocess
import sys
import os

def main():
    print("🏦 Customer Churn Prediction App")
    print("=" * 40)
    
    # Check if we're in the right directory
    required_files = [
        'ann_model.h5',
        'label_encoder_gender.pkl',
        'one_hot_encoder_geography.pkl',
        'scaler.pkl',
        'streamlit_app.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all model files are present in the current directory.")
        return
    
    print("✅ All required files found!")
    print("\nStarting Streamlit application...")
    
    try:
        # Run the streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user.")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()
