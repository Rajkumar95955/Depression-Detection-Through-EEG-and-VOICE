"""
Voice-Based Depression Detection System - GUI Dashboard
Real-time voice recording and analysis with interactive dashboard
"""

import streamlit as st
import os
import warnings
import numpy as np
import pandas as pd
import pickle
import tempfile
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Audio processing
import librosa
import soundfile as sf
import noisereduce as nr
import sounddevice as sd

# ML libraries
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Depression Detection System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================================================

st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .metric-value {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 18px;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Title styling */
    h1 {
        color: white;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    
    h2, h3 {
        color: white;
    }
    
    /* Alert boxes */
    .alert-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .alert-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .alert-danger {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Recording indicator */
    .recording-indicator {
        width: 20px;
        height: 20px;
        background: red;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

class AppConfig:
    """Application configuration"""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    MAX_DURATION = 30
    MIN_DURATION = 3
    
    # Feature extraction parameters
    N_MFCC = 13
    N_CHROMA = 12
    TOP_DB = 20
    
    # Model paths - UPDATE THESE TO YOUR ACTUAL PATHS
    MODEL_PATH = r"C:\Users\RAJ KUMAR\OneDrive\Desktop\Depression Detection Through EEG and VOICE\dataset\output\trained_models"
    SCALER_PATH = r"C:\Users\RAJ KUMAR\OneDrive\Desktop\Depression Detection Through EEG and VOICE\dataset\output\trained_models\scaler.pkl"
    FEATURE_NAMES_PATH = r"C:\Users\RAJ KUMAR\OneDrive\Desktop\Depression Detection Through EEG and VOICE\dataset\output\trained_models\feature_names.pkl"
    
    # Thresholds
    LOW_RISK_THRESHOLD = 0.3
    MEDIUM_RISK_THRESHOLD = 0.6

# ============================================================================
# FEATURE EXTRACTION (Same as training code)
# ============================================================================

def extract_audio_features(audio_data, sr=16000):
    """
    Extract comprehensive audio features from audio data
    """
    try:
        features = {}
        
        # 1. MFCC (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 2. Spectral Features
        spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features['spec_centroid_mean'] = np.mean(spec_centroid)
        features['spec_centroid_std'] = np.std(spec_centroid)
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        features['spec_rolloff_mean'] = np.mean(spec_rolloff)
        features['spec_rolloff_std'] = np.std(spec_rolloff)
        
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
        features['spec_bandwidth_mean'] = np.mean(spec_bandwidth)
        features['spec_bandwidth_std'] = np.std(spec_bandwidth)
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
        
        # 5. Pitch Features
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        pitches = pitches[magnitudes > np.median(magnitudes)]
        pitches = pitches[pitches > 0]
        features['pitch_mean'] = np.mean(pitches) if len(pitches) > 0 else 0
        features['pitch_std'] = np.std(pitches) if len(pitches) > 0 else 0
        
        # 6. RMS Energy
        rms = librosa.feature.rms(y=audio_data)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # 7. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        features['spec_contrast_mean'] = np.mean(contrast)
        features['spec_contrast_std'] = np.std(contrast)
        
        # 8. Temporal Features
        features['duration'] = librosa.get_duration(y=audio_data, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        features['tempo'] = float(tempo)
        
        # 9. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
        features['spec_flatness_mean'] = np.mean(flatness)
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def preprocess_audio(audio_data, sr=16000):
    """
    Preprocess audio: trim, denoise, normalize
    """
    try:
        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=AppConfig.TOP_DB)
        
        # Denoise
        audio_denoised = nr.reduce_noise(y=audio_trimmed, sr=sr, stationary=True)
        
        # Normalize
        max_val = np.max(np.abs(audio_denoised))
        if max_val > 0:
            audio_normalized = audio_denoised / max_val
        else:
            audio_normalized = audio_denoised
        
        return audio_normalized
    except Exception as e:
        st.error(f"Error preprocessing audio: {str(e)}")
        return None

# ============================================================================
# MODEL LOADING AND PREDICTION
# ============================================================================

@st.cache_resource
def load_trained_model():
    """
    Load the trained model, scaler, and feature names
    """
    try:
        # Try to load the best model (you'll need to save this from training)
        model_path = os.path.join(AppConfig.MODEL_PATH, "best_model.pkl")
        scaler_path = AppConfig.SCALER_PATH
        feature_names_path = AppConfig.FEATURE_NAMES_PATH
        
        if not os.path.exists(model_path):
            st.warning("⚠️ No trained model found. Please train the model first using the training script.")
            return None, None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, scaler, feature_names
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_depression(audio_data, model, scaler, feature_names):
    """
    Predict depression from audio data
    """
    try:
        # Preprocess
        audio_processed = preprocess_audio(audio_data, AppConfig.SAMPLE_RATE)
        if audio_processed is None:
            return None, None
        
        # Extract features
        features = extract_audio_features(audio_processed, AppConfig.SAMPLE_RATE)
        if features is None:
            return None, None
        
        # Convert to DataFrame with correct feature order
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[feature_names]  # Ensure correct order
        
        # Handle missing features
        feature_df = feature_df.fillna(0)
        
        # Scale features
        features_scaled = scaler.transform(feature_df)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(features_scaled)[0]
            depression_probability = probability[1]  # Probability of being depressed
        except:
            depression_probability = float(prediction)
        
        return prediction, depression_probability
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# ============================================================================
# AUDIO RECORDING
# ============================================================================

def record_audio(duration, sample_rate=16000):
    """
    Record audio from microphone
    """
    try:
        st.info(f"🎙️ Recording for {duration} seconds...")
        
        # Create placeholder for countdown
        countdown_placeholder = st.empty()
        
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='float32')
        
        # Countdown display
        for i in range(duration, 0, -1):
            countdown_placeholder.markdown(f"### ⏱️ Recording: {i} seconds remaining...")
            time.sleep(1)
        
        sd.wait()  # Wait until recording is finished
        countdown_placeholder.empty()
        
        return audio_data.flatten()
    
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_gauge_chart(value, title):
    """
    Create a gauge chart for depression probability
    """
    # Determine color based on risk level
    if value < AppConfig.LOW_RISK_THRESHOLD:
        color = "#2ecc71"  # Green
        risk_level = "Low Risk"
    elif value < AppConfig.MEDIUM_RISK_THRESHOLD:
        color = "#f39c12"  # Orange
        risk_level = "Medium Risk"
    else:
        color = "#e74c3c"  # Red
        risk_level = "High Risk"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24, 'color': 'white'}},
        number = {'suffix': "%", 'font': {'size': 48, 'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.2)",
            'borderwidth': 3,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(243, 156, 18, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        height=350
    )
    
    return fig, risk_level, color

def create_waveform_plot(audio_data, sr):
    """
    Create waveform visualization
    """
    time_axis = np.linspace(0, len(audio_data) / sr, num=len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        line=dict(color='#667eea', width=1),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)',
        name='Waveform'
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_spectrogram_plot(audio_data, sr):
    """
    Create spectrogram visualization
    """
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        colorscale='Viridis',
        colorbar=dict(title="dB")
    ))
    
    fig.update_layout(
        title="Spectrogram",
        xaxis_title="Time Frame",
        yaxis_title="Frequency Bin",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_feature_importance_plot(features_dict):
    """
    Create feature importance visualization
    """
    # Get top 10 features by absolute value
    sorted_features = sorted(features_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='RdYlGn',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title="Top 10 Audio Features",
        xaxis_title="Feature Value",
        yaxis_title="Feature Name",
        template="plotly_dark",
        height=400,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("<h1>🎙️ Voice-Based Depression Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>AI-powered mental health assessment through voice analysis</p>", unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_trained_model()
    
    if model is None:
        st.error("❌ Please train the model first by running the training script!")
        st.info("📝 Run the training script to generate the required model files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/microphone.png", width=150)
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Choose Recording Method**
           - Record new audio
           - Upload audio file
        
        2. **Speak Clearly**
           - Find a quiet environment
           - Speak naturally for 5-30 seconds
        
        3. **Get Results**
           - View depression probability
           - See detailed analysis
        
        4. **Disclaimer**
           - This is a screening tool only
           - Consult healthcare professionals
           - Not a medical diagnosis
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        duration = st.slider("Recording Duration (seconds)", 
                           min_value=AppConfig.MIN_DURATION, 
                           max_value=AppConfig.MAX_DURATION, 
                           value=10)
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        This system uses machine learning to analyze voice patterns 
        and provide a preliminary assessment of depression indicators.
        
        **Accuracy**: Based on trained model performance
        
        **Features Analyzed**: 60+ acoustic features including pitch, 
        energy, spectral characteristics, and temporal patterns.
        """)
    
    # Main content
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎤 Record Audio", "📁 Upload Audio", "📈 History"])
    
    # Tab 1: Record Audio
    with tab1:
        st.markdown("### 🎙️ Record Your Voice")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🔴 Start Recording", key="record_btn"):
                with st.spinner("Preparing microphone..."):
                    audio_data = record_audio(duration, AppConfig.SAMPLE_RATE)
                
                if audio_data is not None:
                    st.success("✅ Recording completed!")
                    
                    # Save to session state
                    st.session_state['audio_data'] = audio_data
                    st.session_state['sample_rate'] = AppConfig.SAMPLE_RATE
                    
                    # Display audio player
                    st.audio(audio_data, sample_rate=AppConfig.SAMPLE_RATE)
                    
                    # Analyze button
                    if st.button("🔬 Analyze Recording", key="analyze_recorded"):
                        analyze_audio(audio_data, AppConfig.SAMPLE_RATE, model, scaler, feature_names)
    
    # Tab 2: Upload Audio
    with tab2:
        st.markdown("### 📁 Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, FLAC, OGG)",
            type=['wav', 'mp3', 'flac', 'ogg']
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Load audio
            try:
                audio_data, sr = librosa.load(tmp_path, sr=AppConfig.SAMPLE_RATE, mono=True)
                
                st.success("✅ Audio file loaded successfully!")
                st.audio(uploaded_file)
                
                # Display audio info
                duration_sec = len(audio_data) / sr
                st.info(f"**Duration:** {duration_sec:.2f} seconds | **Sample Rate:** {sr} Hz")
                
                # Analyze button
                if st.button("🔬 Analyze Uploaded Audio", key="analyze_uploaded"):
                    analyze_audio(audio_data, sr, model, scaler, feature_names)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error loading audio: {str(e)}")
    
    # Tab 3: History
    with tab3:
        st.markdown("### 📈 Analysis History")
        
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        if len(st.session_state['history']) == 0:
            st.info("No analysis history yet. Record or upload audio to get started!")
        else:
            # Display history as a table
            history_df = pd.DataFrame(st.session_state['history'])
            st.dataframe(history_df, use_container_width=True)
            
            # Plot history trend
            if len(st.session_state['history']) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(st.session_state['history']))),
                    y=[h['Depression Probability (%)'] for h in st.session_state['history']],
                    mode='lines+markers',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title="Depression Probability Trend",
                    xaxis_title="Analysis Number",
                    yaxis_title="Depression Probability (%)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def analyze_audio(audio_data, sr, model, scaler, feature_names):
    """
    Analyze audio and display results
    """
    with st.spinner("🔬 Analyzing audio features..."):
        progress_bar = st.progress(0)
        
        # Step 1: Preprocessing
        progress_bar.progress(20)
        time.sleep(0.3)
        
        # Step 2: Feature extraction
        progress_bar.progress(50)
        time.sleep(0.3)
        
        # Step 3: Prediction
        progress_bar.progress(80)
        prediction, probability = predict_depression(audio_data, model, scaler, feature_names)
        
        progress_bar.progress(100)
        time.sleep(0.2)
        progress_bar.empty()
    
    if prediction is not None:
        st.success("✅ Analysis completed!")
        
        # Create results section
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Main metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart
            gauge_fig, risk_level, risk_color = create_gauge_chart(
                probability, 
                "Depression Probability"
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            # Risk assessment card
            st.markdown(f"""
            <div class="custom-card" style="background: {risk_color}; color: white; height: 350px; display: flex; flex-direction: column; justify-content: center;">
                <h2 style="text-align: center; color: white;">Risk Assessment</h2>
                <h1 style="text-align: center; color: white; font-size: 60px;">{risk_level}</h1>
                <p style="text-align: center; font-size: 24px;">Probability: {probability*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis
        st.markdown("---")
        st.markdown("### 🔍 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Waveform
            waveform_fig = create_waveform_plot(audio_data, sr)
            st.plotly_chart(waveform_fig, use_container_width=True)
        
        with col2:
            # Spectrogram
            spectrogram_fig = create_spectrogram_plot(audio_data, sr)
            st.plotly_chart(spectrogram_fig, use_container_width=True)
        
        # Feature analysis
        st.markdown("---")
        st.markdown("### 📈 Audio Features")
        
        # Extract features for display
        audio_processed = preprocess_audio(audio_data, sr)
        features = extract_audio_features(audio_processed, sr)
        
        if features:
            feature_fig = create_feature_importance_plot(features)
            st.plotly_chart(feature_fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        if probability < AppConfig.LOW_RISK_THRESHOLD:
            st.success("""
            **Low Risk Detected**
            
            Your voice analysis indicates low risk for depression. However:
            - Continue maintaining good mental health practices
            - Stay connected with friends and family
            - Exercise regularly and maintain healthy sleep patterns
            - Seek professional help if you feel you need support
            """)
        elif probability < AppConfig.MEDIUM_RISK_THRESHOLD:
            st.warning("""
            **Medium Risk Detected**
            
            Your voice analysis indicates medium risk. Consider:
            - Talking to a trusted friend or family member
            - Practicing stress-reduction techniques
            - Consulting a mental health professional for assessment
            - Maintaining regular self-care routines
            - Monitoring your emotional state
            """)
        else:
            st.error("""
            **High Risk Detected**
            
            Your voice analysis indicates high risk. Please:
            - **Consult a mental health professional immediately**
            - Reach out to a trusted person for support
            - Contact crisis helplines if needed
            - Remember: This is a screening tool, not a diagnosis
            - Professional evaluation is essential
            
            **Crisis Helplines:**
            - National Suicide Prevention Lifeline: 1-800-273-8255
            - Crisis Text Line: Text HOME to 741741
            """)
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Important Disclaimer**
        
        This tool is designed for screening purposes only and should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare 
        providers for proper evaluation and treatment of mental health concerns.
        """)
        
        # Save to history
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        st.session_state['history'].append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Depression Probability (%)': round(probability * 100, 2),
            'Risk Level': risk_level,
            'Prediction': 'Depressed' if prediction == 1 else 'Not Depressed'
        })

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()