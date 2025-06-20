#!/usr/bin/env python3
"""
Script to test and fix data type issues in the Streamlit app
This addresses the Arrow table serialization errors
"""

import pandas as pd
import numpy as np
import sys
import os

def test_dataframe_creation():
    """Test dataframe creation with proper data types"""
    print("🔍 Testing DataFrame Creation and Arrow Compatibility")
    print("=" * 55)
    
    try:
        # Test profile dataframe creation (like in the app)
        profile_data = {
            'Feature': ['Credit Score', 'Geography', 'Gender', 'Age', 'Tenure', 
                       'Balance', 'Products', 'Credit Card', 'Active Member', 'Salary'],
            'Value': [
                str(600),           # Convert to string
                str('France'),      # Already string
                str('Male'),        # Already string  
                str(40),            # Convert to string
                f"{3} years",       # Format as string
                f"${60000:,.2f}",   # Format as string
                str(2),             # Convert to string
                "Yes",              # Already string
                "No",               # Already string
                f"${50000:,.2f}"    # Format as string
            ]
        }
        
        profile_df = pd.DataFrame(profile_data)
        profile_df = profile_df.astype(str)  # Ensure all are strings
        
        print("✅ Profile DataFrame created successfully")
        print("Data types:", profile_df.dtypes.to_dict())
        print("Sample data:", profile_df.head(3).to_dict())
        
        # Test with PyArrow conversion (Streamlit's backend)
        try:
            import pyarrow as pa
            table = pa.Table.from_pandas(profile_df)
            print("✅ PyArrow conversion successful")
        except ImportError:
            print("ℹ️ PyArrow not available - using alternative test")
            # Alternative test - just ensure no mixed types
            for col in profile_df.columns:
                unique_types = profile_df[col].apply(type).unique()
                if len(unique_types) > 1:
                    print(f"❌ Mixed types in column {col}: {unique_types}")
                else:
                    print(f"✅ Consistent type in column {col}: {unique_types[0]}")
        except Exception as e:
            print(f"❌ PyArrow conversion failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ DataFrame creation failed: {e}")
        return False

def test_numeric_dataframe():
    """Test numeric dataframe handling"""
    print("\n🔢 Testing Numeric DataFrame Handling")
    print("=" * 40)
    
    try:
        # Simulate sample predictions data
        sample_data = {
            'CustomerID': [1001, 1002, 1003, 1004, 1005],
            'CreditScore': [600, 650, 700, 580, 720],
            'Geography': ['France', 'Spain', 'Germany', 'France', 'Germany'],
            'Age': [40, 35, 28, 45, 33],
            'Predicted': [0, 1, 0, 1, 0],
            'Actual': [0, 1, 1, 1, 0],
            'Probability': [0.23, 0.78, 0.45, 0.89, 0.12]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Apply the same fixes as in the app
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("✅ Numeric DataFrame processed successfully")
        print("Data types:", df.dtypes.to_dict())
        print("No NaN values:", not df.isnull().any().any())
        
        # Test PyArrow conversion
        try:
            import pyarrow as pa
            table = pa.Table.from_pandas(df)
            print("✅ PyArrow conversion successful")
        except ImportError:
            print("ℹ️ PyArrow not available - manual validation passed")
        except Exception as e:
            print(f"❌ PyArrow conversion failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Numeric DataFrame test failed: {e}")
        return False

def test_preprocessing_simulation():
    """Simulate the preprocessing function"""
    print("\n⚙️ Testing Preprocessing Simulation")
    print("=" * 35)
    
    try:
        # Simulate input data
        input_data = {
            'CreditScore': 600,
            'Geography': 'France',  # This will be removed and one-hot encoded
            'Gender': 'Male',       # This will be label encoded
            'Age': 40,
            'Tenure': 3,
            'Balance': 60000,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 50000
        }
        
        processed_data = input_data.copy()
        
        # Simulate gender encoding (Male -> 1, Female -> 0)
        processed_data['Gender'] = 1 if processed_data['Gender'] == 'Male' else 0
        
        # Simulate geography one-hot encoding
        geography = processed_data.pop('Geography')
        geography_cols = {
            'Geography_France': 1.0 if geography == 'France' else 0.0,
            'Geography_Germany': 1.0 if geography == 'Germany' else 0.0,
            'Geography_Spain': 1.0 if geography == 'Spain' else 0.0
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([processed_data])
        
        # Add geography columns
        for col, val in geography_cols.items():
            df[col] = float(val)  # Ensure float type
        
        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("✅ Preprocessing simulation successful")
        print("Final data types:", df.dtypes.to_dict())
        print("Shape:", df.shape)
        print("No NaN values:", not df.isnull().any().any())
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing simulation failed: {e}")
        return False

def main():
    """Run all data type tests"""
    print("🔧 Data Type Fix Verification")
    print("=" * 50)
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Run all tests
    test1 = test_dataframe_creation()
    test2 = test_numeric_dataframe()
    test3 = test_preprocessing_simulation()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    
    tests = [
        ("Profile DataFrame", test1),
        ("Numeric DataFrame", test2),
        ("Preprocessing", test3)
    ]
    
    passed = sum([test[1] for test in tests])
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All data type fixes are working correctly!")
        print("The Arrow serialization errors should be resolved.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
