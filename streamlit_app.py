import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    try:
        # Check if files exist
        required_files = {
            'ann_model.h5': 'Trained ANN model',
            'label_encoder_gender.pkl': 'Gender label encoder',
            'one_hot_encoder_geography.pkl': 'Geography one-hot encoder',
            'scaler.pkl': 'Feature scaler'
        }
        
        missing_files = []
        for file, description in required_files.items():
            if not os.path.exists(file):
                missing_files.append(f"{file} ({description})")
        
        if missing_files:
            st.error("❌ Missing required files for prediction:")
            for file in missing_files:
                st.error(f"   • {file}")
            st.info("""
            **Deployment Issue**: Model files are missing from the repository.
            
            **To fix this**:
            1. Ensure model files are not in .gitignore
            2. Add model files to your repository:
               - ann_model.h5
               - label_encoder_gender.pkl
               - one_hot_encoder_geography.pkl  
               - scaler.pkl
            3. Commit and push to your repository
            4. Redeploy to Streamlit Cloud
            """)
            return None, None, None, None        # Load the model with error handling for version compatibility
        model = None
        # Try models in order of preference: Python 3.13 > rebuilt > newer format > original
        model_paths = [
            'ann_model_py313.keras',     # Python 3.13 optimized
            'ann_model_py313.h5',        # Python 3.13 backup
            'ann_model_rebuilt.keras',   # General rebuilt
            'ann_model_rebuilt.h5',      # General rebuilt backup
            'ann_model.keras',           # Converted original
            'ann_model.h5'               # Original
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    st.info(f"🔄 Attempting to load model from {model_path}...")
                    
                    # Use compile=False for better compatibility
                    model = load_model(model_path, compile=False)
                    
                    # Recompile with Python 3.13 compatible settings
                    compile_kwargs = {
                        'optimizer': 'adam',
                        'loss': 'binary_crossentropy',
                        'metrics': ['accuracy']
                    }
                    
                    # Enhanced optimizer for Python 3.13 if available
                    try:
                        if hasattr(tf.keras.optimizers, 'Adam'):
                            compile_kwargs['optimizer'] = tf.keras.optimizers.Adam(
                                learning_rate=0.001,
                                beta_1=0.9,
                                beta_2=0.999,
                                epsilon=1e-7
                            )
                    except:
                        pass  # Fall back to string optimizer
                    
                    model.compile(**compile_kwargs)
                    st.success(f"✅ Model loaded successfully from {model_path}")
                    break
                    
                except Exception as model_error:
                    st.warning(f"⚠️ Failed to load {model_path}: {str(model_error)[:100]}...")
                    continue
        
        if model is None:
            st.error("❌ Failed to load model from any available format")
            st.info(f"""
            **Model Loading Solutions for Python 3.13**:
            
            1. **Create Python 3.13 Compatible Model** (Recommended):
               - Run: `python rebuild_model_py313.py`
               - This creates a model optimized for Python 3.13
            
            2. **Version Information**:
               - Current TensorFlow: {tf.__version__}
               - Python 3.13 requires: TensorFlow >= 2.16.0
               - Recommended: TensorFlow >= 2.17.0
            
            3. **Available Files**: {[f for f in os.listdir('.') if f.endswith(('.h5', '.keras'))]}
            
            **For Python 3.13**: Use the rebuild script to ensure full compatibility
            with the latest Python version and TensorFlow optimizations.
            """)
            return None, None, None, None
        
        # Load the preprocessors
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('one_hot_encoder_geography.pkl', 'rb') as file:
            one_hot_encoder_geography = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        return model, label_encoder_gender, one_hot_encoder_geography, scaler
    
    except Exception as e:
        st.error(f"❌ Error loading model or preprocessors: {e}")
        st.error(f"**Error details**: {str(e)}")
        st.info(f"""
        **Debug Information**:
        - TensorFlow version: {tf.__version__}
        - Current working directory: {os.getcwd()}
        - Available files: {os.listdir('.')}
        
        **Troubleshooting Steps**:
        1. Ensure all model files are present in the deployment
        2. Check TensorFlow version compatibility (currently using {tf.__version__})
        3. Consider re-training and saving the model with the current TensorFlow version
        4. Use native Keras format (.keras) instead of HDF5 (.h5) for better compatibility
        """)
        return None, None, None, None

def preprocess_input(data, label_encoder_gender, one_hot_encoder_geography, scaler):
    """Preprocess input data for prediction with enhanced data type handling"""
    try:
        # Create a copy of the input data
        processed_data = data.copy()
        
        # Encode Gender
        processed_data['Gender'] = label_encoder_gender.transform([processed_data['Gender']])[0]
        
        # One-hot encode Geography
        geography_encoded = one_hot_encoder_geography.transform([[processed_data['Geography']]])
        geography_encoded_df = pd.DataFrame(
            geography_encoded.toarray(), 
            columns=one_hot_encoder_geography.get_feature_names_out(['Geography'])
        )
        
        # Remove Geography from the dictionary
        geography_value = processed_data.pop('Geography')
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([processed_data])
        
        # Add geography encoded columns - ensure they are numeric
        for col in geography_encoded_df.columns:
            df[col] = float(geography_encoded_df[col].iloc[0])
        
        # Ensure all columns are numeric before scaling
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for any NaN values that might have been introduced
        if df.isnull().any().any():
            raise ValueError("NaN values found after preprocessing - check input data types")
        
        # Scale the features
        scaled_data = scaler.transform(df)
        
        return scaled_data
        
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

def main():
    # Title and description
    st.title("🏦 Customer Churn Prediction")
    st.markdown("""
    This application uses an Artificial Neural Network (ANN) to predict whether a bank customer 
    is likely to churn (leave the bank) based on their profile and banking behavior.
    """)
      # Show system information in an expander
    with st.expander("🔧 System Information", expanded=False):
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        st.write(f"**Python Version**: {python_version}")
        st.write(f"**TensorFlow Version**: {tf.__version__}")
        st.write(f"**Streamlit Version**: {st.__version__}")
        
        # Check for Python 3.13 compatibility
        if sys.version_info >= (3, 13):
            st.success("✅ Running on Python 3.13+ - Optimized compatibility")
            st.info("💡 For best performance, ensure you're using the Python 3.13 optimized model")
        elif sys.version_info >= (3, 11):
            st.info("ℹ️ Running on Python 3.11+ - Good compatibility")
        else:
            st.warning("⚠️ Older Python version detected - consider upgrading")
        
        model_files = [f for f in os.listdir('.') if f.endswith(('.h5', '.keras'))]
        st.write(f"**Available Model Files**: {model_files}")
        
        # Highlight Python 3.13 optimized models
        py313_models = [f for f in model_files if 'py313' in f]
        if py313_models:
            st.success(f"🚀 Python 3.13 optimized models available: {py313_models}")
        elif sys.version_info >= (3, 13):
            st.info("💡 Consider running `python rebuild_model_py313.py` for optimal Python 3.13 performance")
    
    # Load model and preprocessors
    model, label_encoder_gender, one_hot_encoder_geography, scaler = load_model_and_preprocessors()
    
    if model is None:
        st.error("Failed to load model or preprocessors. Please check if all required files are present.")
        return
    
    # Sidebar for input
    st.sidebar.header("Customer Information")
    
    # Create input fields
    credit_score = st.sidebar.slider(
        "Credit Score", 
        min_value=300, 
        max_value=850, 
        value=650, 
        help="Customer's credit score (300-850)"
    )
    
    geography = st.sidebar.selectbox(
        "Geography", 
        options=['France', 'Spain', 'Germany'],
        help="Customer's country"
    )
    
    gender = st.sidebar.selectbox(
        "Gender", 
        options=['Male', 'Female'],
        help="Customer's gender"
    )
    
    age = st.sidebar.slider(
        "Age", 
        min_value=18, 
        max_value=100, 
        value=35,
        help="Customer's age"
    )
    
    tenure = st.sidebar.slider(
        "Tenure (Years)", 
        min_value=0, 
        max_value=10, 
        value=5,
        help="Number of years the customer has been with the bank"
    )
    
    balance = st.sidebar.number_input(
        "Account Balance", 
        min_value=0.0, 
        max_value=300000.0, 
        value=100000.0,
        step=1000.0,
        help="Customer's account balance"
    )
    
    num_of_products = st.sidebar.slider(
        "Number of Products", 
        min_value=1, 
        max_value=4, 
        value=2,
        help="Number of bank products the customer has"
    )
    
    has_cr_card = st.sidebar.selectbox(
        "Has Credit Card", 
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether the customer has a credit card"
    )
    
    is_active_member = st.sidebar.selectbox(
        "Is Active Member", 
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether the customer is an active member"
    )
    
    estimated_salary = st.sidebar.number_input(
        "Estimated Salary", 
        min_value=0.0, 
        max_value=200000.0, 
        value=75000.0,
        step=1000.0,
        help="Customer's estimated salary"
    )
    
    # Create prediction button
    if st.sidebar.button("Predict Churn", type="primary"):
        # Prepare input data
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary
        }
        
        try:
            # Preprocess the input
            processed_input = preprocess_input(
                input_data, 
                label_encoder_gender, 
                one_hot_encoder_geography, 
                scaler
            )
              # Make prediction
            prediction_prob = float(model.predict(processed_input)[0][0])
            prediction = 1 if prediction_prob > 0.5 else 0
              # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Customer Profile")
                profile_df = pd.DataFrame({
                    'Feature': ['Credit Score', 'Geography', 'Gender', 'Age', 'Tenure', 
                               'Balance', 'Products', 'Credit Card', 'Active Member', 'Salary'],
                    'Value': [
                        str(credit_score), 
                        str(geography), 
                        str(gender), 
                        str(age), 
                        f"{tenure} years",
                        f"${balance:,.2f}", 
                        str(num_of_products), 
                        "Yes" if has_cr_card else "No",
                        "Yes" if is_active_member else "No",
                        f"${estimated_salary:,.2f}"
                    ]
                })
                
                # Ensure all columns are string type to avoid Arrow conversion issues
                profile_df = profile_df.astype(str)
                st.dataframe(profile_df, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                
                # Display prediction with color coding
                if prediction == 1:
                    st.error(f"🔴 **High Risk of Churn**")
                    st.error(f"Probability: {prediction_prob:.2%}")
                else:
                    st.success(f"🟢 **Low Risk of Churn**")
                    st.success(f"Probability: {prediction_prob:.2%}")
                
                # Create a gauge-like visualization
                st.subheader("Churn Probability")
                progress_value = prediction_prob
                if progress_value > 0.7:
                    color = "🔴"
                elif progress_value > 0.4:
                    color = "🟡"
                else:
                    color = "🟢"
                
                st.metric(
                    label="Churn Risk Level",
                    value=f"{prediction_prob:.2%}",
                    delta=f"{color} {'High' if prediction_prob > 0.5 else 'Low'} Risk"
                )
                
                # Progress bar
                st.progress(progress_value)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    # Additional information
    st.markdown("---")
    st.subheader("About the Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Model Type**: Artificial Neural Network (ANN)
        
        **Architecture**: 
        - Input Layer: 11 features
        - Hidden Layers: 2 layers with ReLU activation
        - Output Layer: 1 neuron with sigmoid activation
        """)
    
    with col2:
        st.info("""
        **Features Used**:
        - Credit Score
        - Geography (One-hot encoded)
        - Gender (Label encoded)
        - Age, Tenure, Balance
        - Number of Products
        - Credit Card Status
        - Active Member Status
        - Estimated Salary
        """)
    
    with col3:
        st.info("""
        **Preprocessing**:
        - Label Encoding for Gender
        - One-Hot Encoding for Geography
        - Standard Scaling for numerical features
        - Feature normalization applied
        """)
      # Model performance section
    st.subheader("Sample Predictions")
    if st.button("Show Sample Predictions"):
        try:
            # Load and display sample predictions
            sample_predictions = pd.read_csv('predictions.csv')
            
            # Ensure proper data types to avoid Arrow conversion issues
            for col in sample_predictions.columns:
                if sample_predictions[col].dtype == 'object':
                    sample_predictions[col] = sample_predictions[col].astype(str)
                else:
                    # Convert numeric columns to float to ensure consistency
                    sample_predictions[col] = pd.to_numeric(sample_predictions[col], errors='coerce')
            
            # Show first 10 predictions
            st.dataframe(sample_predictions.head(10), use_container_width=True)
            
            # Calculate accuracy if Actual and Predicted columns exist
            if 'Actual' in sample_predictions.columns and 'Predicted' in sample_predictions.columns:
                accuracy = (sample_predictions['Actual'] == sample_predictions['Predicted']).mean()
                st.metric("Model Accuracy on Test Set", f"{accuracy:.2%}")
            
        except FileNotFoundError:
            st.warning("Sample predictions file (predictions.csv) not found.")
        except Exception as e:
            st.warning(f"Could not load sample predictions: {str(e)}")

if __name__ == "__main__":
    main()
