# Depression Detection Through EEG and Voice

A multimodal machine learning project that aims to detect signs of depression by analyzing **EEG signals** and **voice/audio features**. This project explores how combining neurological and speech-based signals can improve the accuracy of mental health assessment systems.

---

## 📌 Project Overview

Mental health disorders like depression often go undetected due to the lack of objective assessment tools. This project proposes a **multimodal approach** using:

* **EEG (Electroencephalogram) data** – to capture brain activity patterns
* **Voice data** – to analyze speech tone, pitch, energy, and emotional cues

By fusing these two modalities, the system aims to provide more reliable predictions compared to single-source models.

---

## 🎯 Objectives

* Build a system to classify individuals as **Depressed** or **Not Depressed**
* Extract meaningful features from EEG and audio signals
* Train machine learning / deep learning models for prediction
* Explore multimodal fusion techniques
* Improve accuracy over unimodal approaches

---

## 🧠 Features

* EEG signal preprocessing and feature extraction
* Audio preprocessing (MFCC, pitch, energy, etc.)
* Machine learning / deep learning model training
* Multimodal feature fusion
* Model evaluation using accuracy, precision, recall, F1-score
* Scalable structure for future improvements

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries & Tools:**

  * NumPy, Pandas
  * Scikit-learn
  * Librosa (for audio processing)
  * MNE / SciPy (for EEG processing)
  * TensorFlow / PyTorch (for deep learning models)
  * Matplotlib / Seaborn (for visualization)

---

## 📂 Dataset

This project uses publicly available datasets for:

* EEG-based depression detection
* Speech/audio emotion or depression-related datasets

> Note: Datasets are used strictly for educational and research purposes.

You can place your datasets in the following structure:

```
dataset/
├── eeg/
│   ├── depressed/
│   └── not_depressed/
├── voice/
│   ├── depressed/
│   └── not_depressed/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Depression-Detection-Through-EEG-and-VOICE.git
cd Depression-Detection-Through-EEG-and-VOICE
```

Create virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the main script:

```bash
python main.py
```

Or for individual modules:

```bash
python eeg_model.py
python voice_model.py
python multimodal_fusion.py
```

---

## 🧪 Project Structure

```
Depression-Detection-Through-EEG-and-VOICE/
│
├── .git/
├── .venv/
├── venv/
├── __pycache__/
├── dataset/
│   ├── train/
│   └── test/
├── depression_training.py      # Model training script
├── depression_detection_gui.py # GUI for prediction
├── model_saver.py              # Saves trained models
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── .gitignore                   # Ignored files
```

---

## 📊 Results

* EEG-only model: ~XX% accuracy
* Voice-only model: ~XX% accuracy
* Multimodal (EEG + Voice): ~XX% accuracy

> Results can improve with better datasets, feature engineering, and tuning.

---

## 🔮 Future Scope

* Real-time depression monitoring system
* Mobile or web-based deployment
* Larger and more diverse datasets
* Integration with wearable EEG devices
* Advanced deep learning models (Transformers, Attention models)

---

## 🤝 Contributing

Contributions are welcome!

* Fork the repository
* Create a new branch
* Make your changes
* Submit a pull request

---

## 📜 License

This project is for educational and research purposes only.

---

## 🙌 Acknowledgements

* Open-source datasets and research papers in mental health AI
* Libraries and frameworks used in the project

---

### ⭐ If you find this project helpful, consider giving it a star!
