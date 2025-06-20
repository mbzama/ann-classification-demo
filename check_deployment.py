#!/usr/bin/env python3
"""
Check if required files are properly tracked in git for Streamlit Cloud deployment
"""
import os
import subprocess

def check_git_tracking():
    """Check if model files are tracked in git"""
    print("🔍 Checking Git Tracking for Streamlit Cloud Deployment")
    print("=" * 60)
    
    required_files = [
        'ann_model.h5',
        'label_encoder_gender.pkl', 
        'one_hot_encoder_geography.pkl',
        'scaler.pkl',
        'streamlit_app.py',
        'requirements.txt'
    ]
    
    print("📋 Required Files Status:")
    print("-" * 30)
    
    all_good = True
    
    for file in required_files:
        if os.path.exists(file):
            try:
                # Check if file is tracked in git
                result = subprocess.run(
                    ['git', 'ls-files', file], 
                    capture_output=True, 
                    text=True
                )
                
                if result.stdout.strip():
                    print(f"✅ {file} - EXISTS and TRACKED")
                else:
                    print(f"❌ {file} - EXISTS but NOT TRACKED")
                    all_good = False
            except:
                print(f"⚠️  {file} - EXISTS (git status unknown)")
        else:
            print(f"❌ {file} - NOT FOUND")
            all_good = False
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("🎉 All required files are present and tracked!")
        print("\n📝 Next steps for Streamlit Cloud deployment:")
        print("1. Commit any changes: git add . && git commit -m 'Add model files'")
        print("2. Push to repository: git push")
        print("3. Deploy/redeploy on Streamlit Cloud")
    else:
        print("⚠️  Issues found! Follow these steps:")
        print("\n🔧 To fix:")
        print("1. Add files to git: git add *.h5 *.pkl streamlit_app.py requirements.txt")
        print("2. Commit: git commit -m 'Add model files for deployment'") 
        print("3. Push: git push")
        print("4. Redeploy on Streamlit Cloud")
        
    print("\n💡 Note: Make sure .gitignore doesn't exclude .h5 or .pkl files!")

if __name__ == "__main__":
    check_git_tracking()
