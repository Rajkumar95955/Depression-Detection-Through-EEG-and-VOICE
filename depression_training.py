"""
Voice-Based Depression Detection System
Automated training from organized dataset folders
Dataset structure: dataset/train/depressed/ and dataset/train/not_depressed/
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Audio processing libraries
import librosa
import soundfile as sf
import noisereduce as nr

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('default')

# ============================================================================
# CONFIGURATION - ONLY CHANGE THIS PATH
# ============================================================================

class Config:
    """Configuration parameters - Set your dataset path here"""
    
    # ⭐ MAIN DATASET PATH - CHANGE THIS TO YOUR FOLDER PATH ⭐
    DATASET_PATH = r"C:\Users\RAJ KUMAR\OneDrive\Desktop\Depression Detection Through EEG and VOICE\dataset"
    
    # Auto-generated paths (don't change these)
    TRAIN_PATH = os.path.join(DATASET_PATH, "train")
    DEPRESSED_PATH = os.path.join(TRAIN_PATH, "depressed")
    NOT_DEPRESSED_PATH = os.path.join(TRAIN_PATH, "not_depressed")
    
    # Output paths
    OUTPUT_PATH = os.path.join(DATASET_PATH, "output")
    CLEANED_PATH = os.path.join(OUTPUT_PATH, "cleaned_audio")
    FEATURES_CSV = os.path.join(OUTPUT_PATH, "extracted_features.csv")
    MODELS_PATH = os.path.join(OUTPUT_PATH, "trained_models")
    PLOTS_PATH = os.path.join(OUTPUT_PATH, "plots")
    
    # Audio parameters
    SAMPLE_RATE = 16000
    TOP_DB = 20
    
    # Feature extraction parameters
    N_MFCC = 13
    N_CHROMA = 12
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_output_directories():
    """Create necessary output directories"""
    config = Config()
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    os.makedirs(config.CLEANED_PATH, exist_ok=True)
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    os.makedirs(config.PLOTS_PATH, exist_ok=True)
    os.makedirs(os.path.join(config.CLEANED_PATH, "depressed"), exist_ok=True)
    os.makedirs(os.path.join(config.CLEANED_PATH, "not_depressed"), exist_ok=True)

def validate_dataset_structure():
    """Validate that the dataset has the correct structure"""
    config = Config()
    
    print("\n" + "="*80)
    print("VALIDATING DATASET STRUCTURE")
    print("="*80)
    
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"❌ Dataset path not found: {config.DATASET_PATH}")
    
    if not os.path.exists(config.TRAIN_PATH):
        raise FileNotFoundError(f"❌ Train folder not found: {config.TRAIN_PATH}")
    
    if not os.path.exists(config.DEPRESSED_PATH):
        raise FileNotFoundError(f"❌ Depressed folder not found: {config.DEPRESSED_PATH}")
    
    if not os.path.exists(config.NOT_DEPRESSED_PATH):
        raise FileNotFoundError(f"❌ Not depressed folder not found: {config.NOT_DEPRESSED_PATH}")
    
    depressed_files = [f for f in os.listdir(config.DEPRESSED_PATH) 
                       if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    not_depressed_files = [f for f in os.listdir(config.NOT_DEPRESSED_PATH) 
                           if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    
    print(f"✅ Dataset structure validated!")
    print(f"📊 Depressed samples: {len(depressed_files)}")
    print(f"📊 Not depressed samples: {len(not_depressed_files)}")
    print(f"📊 Total samples: {len(depressed_files) + len(not_depressed_files)}")
    
    if len(depressed_files) == 0 or len(not_depressed_files) == 0:
        raise ValueError("❌ One or both folders are empty!")
    
    return len(depressed_files), len(not_depressed_files)

# ============================================================================
# STEP 1: AUDIO CLEANING
# ============================================================================

def clean_audio_file(input_path, output_path, sr=16000, top_db=20):
    """Clean a single audio file: load, trim silence, denoise, normalize"""
    try:
        y, sr = librosa.load(input_path, sr=sr, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        
        if len(y_trimmed) < sr * 0.5:
            return False, "Audio too short after trimming"
        
        y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr, stationary=True)
        
        max_val = np.max(np.abs(y_denoised))
        if max_val > 0:
            y_normalized = y_denoised / max_val
        else:
            return False, "Silent audio file"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y_normalized, sr)
        
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def clean_all_audio_files():
    """Clean all audio files from both depressed and not_depressed folders"""
    config = Config()
    
    print("\n" + "="*80)
    print("STEP 1: CLEANING AUDIO FILES")
    print("="*80)
    
    total_cleaned = 0
    total_errors = 0
    
    categories = {
        'depressed': (config.DEPRESSED_PATH, os.path.join(config.CLEANED_PATH, "depressed")),
        'not_depressed': (config.NOT_DEPRESSED_PATH, os.path.join(config.CLEANED_PATH, "not_depressed"))
    }
    
    for category, (input_folder, output_folder) in categories.items():
        print(f"\n🔄 Processing {category} files...")
        
        audio_files = [f for f in os.listdir(input_folder) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        for filename in audio_files:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.mp3', '.wav').replace('.flac', '.wav').replace('.ogg', '.wav'))
            
            success, message = clean_audio_file(input_path, output_path, config.SAMPLE_RATE, config.TOP_DB)
            
            if success:
                print(f"  ✅ {filename}")
                total_cleaned += 1
            else:
                print(f"  ❌ {filename}: {message}")
                total_errors += 1
    
    print(f"\n📊 Summary: {total_cleaned} cleaned, {total_errors} errors")
    return total_cleaned

# ============================================================================
# STEP 2: FEATURE EXTRACTION
# ============================================================================

def extract_audio_features(file_path, sr=16000):
    """Extract comprehensive audio features from a WAV file"""
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        features = {}
        
        # 1. MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 2. Spectral Features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spec_centroid_mean'] = np.mean(spec_centroid)
        features['spec_centroid_std'] = np.std(spec_centroid)
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spec_rolloff_mean'] = np.mean(spec_rolloff)
        features['spec_rolloff_std'] = np.std(spec_rolloff)
        
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spec_bandwidth_mean'] = np.mean(spec_bandwidth)
        features['spec_bandwidth_std'] = np.std(spec_bandwidth)
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        
        # 5. Pitch Features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitches = pitches[magnitudes > np.median(magnitudes)]
        pitches = pitches[pitches > 0]
        features['pitch_mean'] = np.mean(pitches) if len(pitches) > 0 else 0
        features['pitch_std'] = np.std(pitches) if len(pitches) > 0 else 0
        
        # 6. RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 7. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spec_contrast_mean'] = np.mean(contrast)
        features['spec_contrast_std'] = np.std(contrast)
        
        # 8. Temporal Features
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # 9. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spec_flatness_mean'] = np.mean(flatness)
        
        return features
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return None

def extract_features_from_dataset():
    """Extract features from all cleaned audio files"""
    config = Config()
    
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING AUDIO FEATURES")
    print("="*80)
    
    all_features = []
    
    categories = {
        'depressed': (os.path.join(config.CLEANED_PATH, "depressed"), 1),
        'not_depressed': (os.path.join(config.CLEANED_PATH, "not_depressed"), 0)
    }
    
    for category, (folder_path, label) in categories.items():
        print(f"\n🔄 Extracting features from {category} files...")
        
        audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        for filename in audio_files:
            file_path = os.path.join(folder_path, filename)
            features = extract_audio_features(file_path, config.SAMPLE_RATE)
            
            if features:
                features['file_name'] = filename
                features['label'] = label
                features['category'] = category
                all_features.append(features)
                print(f"  ✅ {filename}")
            else:
                print(f"  ⚠️ Skipped: {filename}")
    
    if not all_features:
        raise ValueError("❌ No features extracted!")
    
    df = pd.DataFrame(all_features)
    df.to_csv(config.FEATURES_CSV, index=False)
    
    print(f"\n✅ Features saved to: {config.FEATURES_CSV}")
    print(f"📊 Total samples: {len(df)}")
    print(f"📊 Features per sample: {len(df.columns) - 3}")
    print(f"\n📊 Class Distribution:")
    print(df['category'].value_counts())
    
    return df

# ============================================================================
# STEP 3: MODEL TRAINING
# ============================================================================

def prepare_data(df):
    """Prepare data for training"""
    config = Config()
    
    print("\n" + "="*80)
    print("STEP 3: PREPARING DATA FOR TRAINING")
    print("="*80)
    
    X = df.drop(columns=['file_name', 'label', 'category'])
    y = df['label']
    
    if X.isnull().sum().sum() > 0:
        print("⚠️  Found missing values. Filling with median...")
        X = X.fillna(X.median())
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"✅ Feature matrix shape: {X.shape}")
    print(f"✅ Class distribution:")
    print(f"   - Not Depressed (0): {(y == 0).sum()}")
    print(f"   - Depressed (1): {(y == 1).sum()}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, stratify=y
    )
    
    print(f"\n✅ Train set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        y_pred_proba = None
        roc_auc = None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"\n📊 {model_name} Results:")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    if roc_auc:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Depressed', 'Depressed']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📉 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Not Dep  Depressed")
    print(f"Actual Not Dep      {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"       Depressed    {cm[1][0]:3d}      {cm[1][1]:3d}")
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }

def train_all_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    print("\n" + "="*80)
    print("STEP 4: TRAINING MULTIPLE MODELS")
    print("="*80)
    
    models = {
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                                min_samples_split=5, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski'),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        result = train_and_evaluate_model(model, name, X_train, X_test, y_train, y_test)
        results.append(result)
    
    return results

# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================

def plot_model_comparison(results):
    """Plot comprehensive comparison of model performance"""
    config = Config()
    
    print("\n" + "="*80)
    print("STEP 5: VISUALIZING RESULTS")
    print("="*80)
    
    models = [r['name'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    precisions = [r['precision'] * 100 for r in results]
    recalls = [r['recall'] * 100 for r in results]
    f1_scores = [r['f1'] * 100 for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    bars = axes[0, 0].bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    x = np.arange(len(models))
    width = 0.2
    axes[0, 1].bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3498db')
    axes[0, 1].bar(x - 0.5*width, precisions, width, label='Precision', color='#2ecc71')
    axes[0, 1].bar(x + 0.5*width, recalls, width, label='Recall', color='#e74c3c')
    axes[0, 1].bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#f39c12')
    axes[0, 1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Score (%)', fontsize=11)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=15)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    
    axes[1, 0].scatter(recalls, precisions, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    for i, model in enumerate(models):
        axes[1, 0].annotate(model, (recalls[i], precisions[i]), 
                           fontsize=9, ha='center', va='bottom')
    axes[1, 0].set_xlabel('Recall (%)', fontsize=11)
    axes[1, 0].set_ylabel('Precision (%)', fontsize=11)
    axes[1, 0].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    bars = axes[1, 1].barh(models, f1_scores, color=colors, alpha=0.8, edgecolor='black')
    for bar, f1 in zip(bars, f1_scores):
        axes[1, 1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{f1:.2f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel('F1-Score (%)', fontsize=11)
    axes[1, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_PATH, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")

def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models"""
    config = Config()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    cmaps = ['Reds', 'Blues', 'Greens', 'Oranges']
    
    for idx, result in enumerate(results):
        cm = result['confusion_matrix']
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        im = axes[idx].imshow(cm, cmap=cmaps[idx], aspect='auto', alpha=0.8)
        plt.colorbar(im, ax=axes[idx], label='Count')
        
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                             ha='center', va='center', fontsize=12, fontweight='bold',
                             color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Not Depressed', 'Depressed'])
        axes[idx].set_yticklabels(['Not Depressed', 'Depressed'])
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_title(f"{result['name']}\nAccuracy: {result['accuracy']*100:.2f}%",
                          fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_PATH, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    config = Config()
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for idx, result in enumerate(results):
        if result['pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['pred_proba'])
            auc = result['roc_auc']
            plt.plot(fpr, tpr, color=colors[idx], linewidth=2, 
                    label=f"{result['name']} (AUC = {auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(config.PLOTS_PATH, 'roc_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")

def plot_learning_curve(estimator, title, X, y, cv=5):
    """Plot learning curve for a model"""
    config = Config()
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', shuffle=True, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='#e74c3c', linewidth=2, label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='#2ecc71', linewidth=2, label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='#e74c3c')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                     alpha=0.15, color='#2ecc71')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(config.PLOTS_PATH, f'learning_curve_{title.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")

def generate_summary_report(results, n_depressed, n_not_depressed):
    """Generate a summary report of all models"""
    best_model = max(results, key=lambda x: x['accuracy'])
    summary_path = os.path.join(Config.OUTPUT_PATH, "training_summary.txt")
    
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("VOICE-BASED DEPRESSION DETECTION SYSTEM\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total depressed samples: {n_depressed}\n")
            f.write(f"Total non-depressed samples: {n_not_depressed}\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            for res in results:
                f.write(f"- {res['name']}: Accuracy={res['accuracy']*100:.2f}%, F1={res['f1']*100:.2f}%\n")
            
            f.write("\n🏆 BEST PERFORMING MODEL: " + best_model['name'] + "\n")
            f.write(f"Accuracy: {best_model['accuracy']*100:.2f}%\n")
            f.write(f"F1-Score: {best_model['f1']*100:.2f}%\n\n")
            
            f.write("OUTPUTS SAVED IN:\n")
            f.write(f"- Cleaned audio files: {Config.CLEANED_PATH}\n")
            f.write(f"- Extracted features: {Config.FEATURES_CSV}\n")
            f.write(f"- Visualization plots: {Config.PLOTS_PATH}\n")
            f.write(f"- Trained models: {Config.MODELS_PATH}\n")
            
        print(f"\nSummary report saved to: {summary_path}")
        
    except Exception as e:
        print(f"\nCould not write summary report: {e}")

# ============================================================================
# MODEL SAVING FUNCTIONS - MUST BE DEFINED BEFORE main()
# ============================================================================

def save_all_models(results, config):
    """Save all trained models for backup"""
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

def save_model_artifacts(best_model_result, scaler, feature_names, config):
    """Save the best model, scaler, and feature names for GUI usage"""
    print("\n" + "="*80)
    print("SAVING MODEL ARTIFACTS FOR GUI")
    print("="*80)
    
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
        
        print("\n" + "="*80)
        print("✅ ALL MODEL ARTIFACTS SAVED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Model files location: {config.MODELS_PATH}")
        print(f"   - best_model.pkl")
        print(f"   - scaler.pkl")
        print(f"   - feature_names.pkl")
        print(f"   - model_metadata.pkl")
        print(f"\n🎯 These files are ready for the GUI application!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error saving model artifacts: {str(e)}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete pipeline with model saving"""
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
        
        # Step 6: Save models for GUI
        print("\n" + "="*80)
        print("PREPARING MODELS FOR PRODUCTION (GUI)")
        print("="*80)
        
        save_all_models(results, Config)
        best_model_result = max(results, key=lambda x: x['accuracy'])
        success = save_model_artifacts(best_model_result, scaler, feature_names, Config)
        
        if success:
            print("\n🎉 Models are ready for the GUI application!")
            print(f"\n📋 Next Steps:")
            print(f"   1. Install GUI dependencies: pip install streamlit sounddevice plotly")
            print(f"   2. Run the GUI: streamlit run depression_detection_gui.py")
            print(f"   3. Update GUI paths to: {Config.MODELS_PATH}")
        
        # Step 7: Final report
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

if __name__ == "__main__":
    main()









