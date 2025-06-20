# Streamlit Deployment Fix Guide

## Issue
Getting error: `Error loading model or preprocessors: Error when deserializing class 'InputLayer'` in Streamlit Cloud deployment with Python 3.11.0.

## Root Cause
This error typically occurs due to:
1. **Version Mismatch**: TensorFlow version used to save the model differs from the one used to load it
2. **Legacy Format**: HDF5 format (.h5) has compatibility issues across TensorFlow versions
3. **Python 3.11 Compatibility**: Some TensorFlow versions have issues with Python 3.11

## Solutions Implemented

### 1. Updated Requirements.txt
```
tensorflow==2.15.0  # Python 3.11 compatible version
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
streamlit==1.28.1
protobuf==3.20.3
```

### 2. Enhanced Model Loading
The Streamlit app now:
- Tries to load from `.keras` format first (more compatible)
- Falls back to `.h5` format if needed
- Loads with `compile=False` and recompiles to avoid metric issues
- Provides detailed error messages and troubleshooting tips

### 3. Conversion Script
Use `convert_model.py` to convert your existing model to the newer format:
```bash
python convert_model.py
```

## Deployment Steps

### Option A: Convert Model (Recommended)
1. Run the conversion script locally:
   ```bash
   python convert_model.py
   ```
2. Add the new `ann_model.keras` file to your repository
3. Commit and push changes
4. Redeploy on Streamlit Cloud

### Option B: Update Requirements Only
1. Use the updated `requirements.txt`
2. Commit and push changes
3. Redeploy on Streamlit Cloud

## Testing Locally
Before deploying, test locally:
```bash
# Install requirements
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

## Additional Notes
- The app now shows system information to help debug version issues
- Enhanced error messages provide specific troubleshooting steps
- Model loading is more robust with fallback mechanisms

## If Issues Persist
1. Check Streamlit Cloud logs for detailed error messages
2. Ensure all model files are committed to the repository
3. Verify that files are not listed in `.gitignore`
4. Consider re-training the model with the target TensorFlow version
