"""
Model Saver Extension - Add this to your training script
Saves trained models, scaler, and feature names for GUI usage

ADD THESE FUNCTIONS TO YOUR TRAINING SCRIPT
"""

import pickle
import os
import joblib

# ============================================================================
# ADD THESE FUNCTIONS TO YOUR TRAINING SCRIPT
# ============================================================================

def save_model_artifacts(best_model_result, scaler, feature_names, config):
    """
    Save the best model, scaler, and feature names for production use
    
    Args:
        best_model_result: Dictionary containing the best model
        scaler: StandardScaler object used for feature scaling
        feature_names: List of feature names in correct order
        config: Config object with paths
    """
    print("\n" + "="*80)
    print("SAVING MODEL ARTIFACTS FOR PRODUCTION")
    print("="*80)
    
    # Ensure models directory exists
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    
    try:
        # 1. Save the best model
        model_path = os.path.join(config.MODELS_PATH, "best_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_model_result['model'], f)
        print(f"✅ Saved best model: {model_path}")
        print(f"   Model type: {best_model_result['name']}")
        print(f"   Accuracy: {best_model_result['accuracy']*100:.2f}%")
        
        # 2. Save the scaler
        scaler_path = os.path.join(config.MODELS_PATH, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Saved scaler: {scaler_path}")
        
        # 3. Save feature names
        feature_names_path = os.path.join(config.MODELS_PATH, "feature_names.pkl")
        with open(feature_names_path, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"✅ Saved feature names: {feature_names_path}")
        print(f"   Total features: {len(feature_names)}")
        
        # 4. Save model metadata
        metadata = {
            'model_name': best_model_result['name'],
            'accuracy': best_model_result['accuracy'],
            'precision': best_model_result['precision'],
            'recall': best_model_result['recall'],
            'f1_score': best_model_result['f1'],
            'roc_auc': best_model_result['roc_auc'],
            'feature_count': len(feature_names),
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sample_rate': config.SAMPLE_RATE,
            'n_mfcc': config.N_MFCC,
            'n_chroma': config.N_CHROMA
        }
        
        metadata_path = os.path.join(config.MODELS_PATH, "model_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Saved metadata: {metadata_path}")
        
        # 5. Save all trained models (optional - for comparison)
        print(f"\n📦 Saving all trained models for backup...")
        
        # This will be populated in the modified main function
        # Just creating the structure here
        
        print("\n" + "="*80)
        print("✅ ALL MODEL ARTIFACTS SAVED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Model files location: {config.MODELS_PATH}")
        print(f"   - best_model.pkl")
        print(f"   - scaler.pkl")
        print(f"   - feature_names.pkl")
        print(f"   - model_metadata.pkl")
        print(f"\n🎯 These files are ready to be used by the GUI application!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error saving model artifacts: {str(e)}")
        return False


def save_all_models(results, config):
    """
    Save all trained models for later comparison
    
    Args:
        results: List of result dictionaries from train_all_models
        config: Config object with paths
    """
    print(f"\n💾 Saving all trained models...")
    
    for result in results:
        model_name = result['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(config.MODELS_PATH, f"{model_name}_model.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
            print(f"   ✅ Saved: {model_name}_model.pkl (Accuracy: {result['accuracy']*100:.2f}%)")
        except Exception as e:
            print(f"   ❌ Error saving {model_name}: {str(e)}")


# ============================================================================
# MODIFIED MAIN FUNCTION - REPLACE YOUR EXISTING main() FUNCTION WITH THIS
# ============================================================================

def main():
    """
    Main function to run the complete pipeline with model saving
    """
    try:
        print("🚀 VOICE-BASED DEPRESSION DETECTION SYSTEM")
        print("=" * 80)
        
        # Step 0: Setup and validation
        create_output_directories()
        n_depressed, n_not_depressed = validate_dataset_structure()
        
        # Step 1: Audio cleaning
        total_cleaned = clean_all_audio_files()
        if total_cleaned == 0:
            raise ValueError("❌ No audio files were successfully cleaned!")
        
        # Step 2: Feature extraction
        df = extract_features_from_dataset()
        
        # Step 3: Model training
        X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
        results = train_all_models(X_train, X_test, y_train, y_test)
        
        # Step 4: Visualization
        plot_model_comparison(results)
        plot_confusion_matrices(results, y_test)
        plot_roc_curves(results, y_test)
        
        # Plot learning curve for best model
        best_model = max(results, key=lambda x: x['accuracy'])
        plot_learning_curve(best_model['model'], f"{best_model['name']} Learning Curve", 
                           X_train, y_train, Config.CV_FOLDS)
        
        # Step 5: Generate summary
        generate_summary_report(results, n_depressed, n_not_depressed)
        
        # ⭐ NEW STEP 6: Save model artifacts for GUI
        print("\n" + "="*80)
        print("PREPARING MODELS FOR PRODUCTION (GUI)")
        print("="*80)
        
        # Save all models for backup
        save_all_models(results, Config)
        
        # Save best model and required artifacts
        best_model_result = max(results, key=lambda x: x['accuracy'])
        success = save_model_artifacts(best_model_result, scaler, feature_names, Config)
        
        if success:
            print("\n🎉 Models are ready for the GUI application!")
            print(f"\n📋 Next Steps:")
            print(f"   1. Install GUI dependencies: pip install streamlit sounddevice plotly")
            print(f"   2. Run the GUI: streamlit run depression_detection_gui.py")
            print(f"   3. Ensure the GUI script can access: {Config.MODELS_PATH}")
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\n🏆 BEST MODEL: {best_result['name']}")
        print(f"📊 ACCURACY: {best_result['accuracy']*100:.2f}%")
        print(f"📈 F1-SCORE: {best_result['f1']*100:.2f}%")
        
        print(f"\n📁 All outputs saved in: {Config.OUTPUT_PATH}")
        print("\n📋 Generated files:")
        print(f"   ✅ Cleaned audio files: {Config.CLEANED_PATH}")
        print(f"   ✅ Extracted features: {Config.FEATURES_CSV}")
        print(f"   ✅ Visualization plots: {Config.PLOTS_PATH}")
        print(f"   ✅ Trained models: {Config.MODELS_PATH}")
        print(f"   ✅ Training summary: {os.path.join(Config.OUTPUT_PATH, 'training_summary.txt')}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check that your dataset path is correct")
        print("   2. Ensure audio files are in supported formats (wav, mp3, flac, ogg)")
        print("   3. Verify folder structure: dataset/train/depressed/ and dataset/train/not_depressed/")
        print("   4. Check that audio files are not corrupted")
        raise


# ============================================================================
# QUICK TEST FUNCTION - Test if saved models work
# ============================================================================

def test_saved_models():
    """
    Test if the saved models can be loaded and used
    """
    print("\n" + "="*80)
    print("TESTING SAVED MODELS")
    print("="*80)
    
    config = Config()
    
    try:
        # Load model
        model_path = os.path.join(config.MODELS_PATH, "best_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        
        # Load scaler
        scaler_path = os.path.join(config.MODELS_PATH, "scaler.pkl")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully")
        
        # Load feature names
        feature_names_path = os.path.join(config.MODELS_PATH, "feature_names.pkl")
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✅ Feature names loaded successfully ({len(feature_names)} features)")
        
        # Load metadata
        metadata_path = os.path.join(config.MODELS_PATH, "model_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print("✅ Metadata loaded successfully")
        
        print("\n📊 Model Information:")
        print(f"   Model: {metadata['model_name']}")
        print(f"   Accuracy: {metadata['accuracy']*100:.2f}%")
        print(f"   F1-Score: {metadata['f1_score']*100:.2f}%")
        print(f"   Training Date: {metadata['training_date']}")
        print(f"   Features: {metadata['feature_count']}")
        
        print("\n" + "="*80)
        print("✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("🚀 Ready for GUI deployment!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing models: {str(e)}")
        return False


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
HOW TO USE THIS EXTENSION:

1. INTEGRATE WITH YOUR TRAINING SCRIPT:
   
   Option A - Add to existing script:
   - Copy the save_model_artifacts() and save_all_models() functions
   - Replace your main() function with the modified version above
   - Keep all your existing functions (they work as-is)
   
   Option B - Create new file:
   - Save this as "model_saver.py" in the same directory
   - Import in your training script: from model_saver import save_model_artifacts, save_all_models
   - Add the save calls at the end of your main() function

2. RUN THE TRAINING SCRIPT:
   python your_training_script.py
   
   This will now create in dataset/output/trained_models/:
   - best_model.pkl (the best performing model)
   - scaler.pkl (feature scaler)
   - feature_names.pkl (list of feature names)
   - model_metadata.pkl (training information)
   - individual model files (svm_rbf_model.pkl, etc.)

3. VERIFY MODELS WERE SAVED:
   - Add this to the end of your script:
     if __name__ == "__main__":
         main()
         test_saved_models()  # This will verify everything works

4. UPDATE GUI CONFIGURATION:
   In the GUI script (depression_detection_gui.py), update the paths:
   
   class AppConfig:
       MODEL_PATH = "dataset/output/trained_models"
       SCALER_PATH = "dataset/output/trained_models/scaler.pkl"
       FEATURE_NAMES_PATH = "dataset/output/trained_models/feature_names.pkl"

5. RUN THE GUI:
   streamlit run depression_detection_gui.py

THAT'S IT! Your GUI will now use the trained models.
"""

# ============================================================================
# EXAMPLE: Complete integration snippet for your training script
# ============================================================================

"""
# At the top of your training script, add:
import pickle

# Replace your main() function with the modified version above
# Or add these lines at the end of your existing main() function:

def main():
    # ... your existing code ...
    
    # After: results = train_all_models(X_train, X_test, y_train, y_test)
    
    # Add these lines:
    save_all_models(results, Config)
    best_model_result = max(results, key=lambda x: x['accuracy'])
    save_model_artifacts(best_model_result, scaler, feature_names, Config)
    
    # ... rest of your code ...

if __name__ == "__main__":
    main()
    test_saved_models()  # Optional: verify models work
"""