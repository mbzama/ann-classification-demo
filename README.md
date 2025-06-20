# Customer Churn Prediction - Streamlit App

This Streamlit application demonstrates customer churn prediction using an Artificial Neural Network (ANN) trained on banking customer data.

## Features

- 🎯 **Interactive Prediction**: Input customer details and get real-time churn probability
- 📊 **Visual Results**: Color-coded risk levels and probability metrics
- 🔧 **Complete Pipeline**: Includes data preprocessing with the same transformations used during training
- 📈 **Model Information**: Displays model architecture and feature details
- 📋 **Sample Predictions**: View sample predictions from the test set

## Required Files

Make sure these files are present in the project directory:

- `ann_model.h5` - Trained ANN model
- `label_encoder_gender.pkl` - Gender label encoder
- `one_hot_encoder_geography.pkl` - Geography one-hot encoder  
- `scaler.pkl` - Feature scaler
- `predictions.csv` - Sample predictions (optional)
- `streamlit_app.py` - Main Streamlit application

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the App

### Option 1: Using Python script
```bash
python run_app.py
```

### Option 2: Using Streamlit directly
```bash
streamlit run streamlit_app.py
```

### Option 3: Using batch file (Windows)
```bash
run_app.bat
```

## How to Use

1. **Launch the app** using one of the methods above
2. **Input customer details** in the sidebar:
   - Credit Score (300-850)
   - Geography (France, Spain, Germany)
   - Gender (Male, Female)
   - Age (18-100)
   - Tenure with bank (0-10 years)
   - Account Balance
   - Number of products (1-4)
   - Credit card status
   - Active member status
   - Estimated salary

3. **Click "Predict Churn"** to get results
4. **View the prediction**:
   - 🟢 Green: Low churn risk (probability < 50%)
   - 🔴 Red: High churn risk (probability ≥ 50%)

## Model Details

### Architecture
- **Input Layer**: 11 features (after preprocessing)
- **Hidden Layers**: 2 layers with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation for binary classification

### Features Used
- Credit Score
- Geography (One-hot encoded: France, Spain, Germany)
- Gender (Label encoded: Male=1, Female=0)
- Age
- Tenure (years with bank)
- Account Balance
- Number of Products
- Has Credit Card (0/1)
- Is Active Member (0/1)
- Estimated Salary

### Preprocessing Pipeline
1. **Drop unnecessary columns**: RowNumber, CustomerId, Surname
2. **Label encode Gender**: Male=1, Female=0
3. **One-hot encode Geography**: Creates binary columns for each country
4. **Standard scaling**: Normalize all numerical features

## Understanding the Results

### Risk Levels
- **Low Risk (🟢)**: Probability < 50% - Customer likely to stay
- **High Risk (🔴)**: Probability ≥ 50% - Customer likely to churn

### Churn Probability
The model outputs a probability score between 0 and 1:
- **0.0 - 0.3**: Very low risk
- **0.3 - 0.5**: Low risk  
- **0.5 - 0.7**: Moderate risk
- **0.7 - 1.0**: High risk

## Business Use Cases

This application can be used to:

1. **Identify at-risk customers** for targeted retention campaigns
2. **Analyze customer segments** with high churn probability
3. **Test different customer profiles** to understand risk factors
4. **Support decision making** for customer relationship management

## Technical Stack

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Model Type**: Artificial Neural Network (ANN)

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all packages are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Missing files**: Ensure all pickle files and the model file are present

3. **Path issues**: Run the app from the project directory containing all files

4. **Port conflicts**: If port 8501 is busy, Streamlit will suggest an alternative

### Error Messages

- "Error loading model or preprocessors": Check if all .pkl and .h5 files exist
- "Error making prediction": Usually indicates preprocessing issues

## Next Steps

You can extend this application by:

- Adding batch prediction capabilities
- Including model performance metrics
- Adding data visualization features
- Implementing model retraining functionality
- Adding export capabilities for predictions

---

*This application demonstrates the practical deployment of machine learning models using Streamlit for interactive web applications.*
