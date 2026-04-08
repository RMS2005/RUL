# Predictive Maintenance: Remaining Useful Life (RUL) Prediction

## 📖 Overview
This repository contains the comprehensive implementation codebase for predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines. Centered around predictive maintenance for the aerospace industry, the project applies a wide array of Machine Learning (ML) techniques and advanced Deep Learning (DL) sequence models (like Bidirectional LSTMs and CNN-Transformers). 

The ultimate goal is to forecast the exact time until an engine is likely to fail, enabling optimized, proactive maintenance schedules, preventing catastrophic failures, and drastically improving overall aviation safety.

## 🗄️ Dataset
This project heavily utilizes the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. 
- The dataset consists of multiple simulated time-series trajectories for turbofan engines.
- It records various sensor readings (temperature, pressure, fan speed, etc.) and operating settings from standard baseline operation all the way to complete engine failure.
- A critical challenge addressed in this dataset is **covariate shift** and the variance in engine sensor values over time.

## 🛠️ Extensive Methodology

### 1. Data Preprocessing & Cleaning
- **Constant Sensor Removal**: Dropped sensor columns that showed zero variance across the lifecycles, as they introduce noise and carry no predictive power.
- **Covariate Shift Detection**: Executed feature-wise train-test distribution diagnostics to prevent extrapolation risks.
- **MinMax Scaling & Normalization**: Due to operating settings varying drastically, features were uniformly scaled for improved neural network convergence.

### 2. Advanced Feature Engineering
- **RUL Clipping (Winsorization at 125 cycles)**: Implementing piece-wise linear degradation models. The RUL is capped at 125 cycles during the engine's early "healthy" phase to prevent the model from noise-fitting and force it to learn only the actual degradation dynamics.
- **Rolling Statistics**: Transformed raw sensor time series into localized trend, variability, and extremal features.
- **Cyclic Normalization (Differences/Rate of Change)**: Extracted relative sequence positions to encode rate and direction of change.
- **Sliding Window Sequencing**: Formatted 2D tabular data into 3D sequential overlaps (e.g., `(num_batches, window_length, num_features)`) for robust sequence modeling.

## 🧠 Models Implemented

### Machine Learning Baselines & Classical Approaches
Extensive baseline testing was carried out to establish performance benchmarks:
- **Linear Regression**
- **Support Vector Regressor (SVR) & Support Vector Classifier (SVC)**
- **Random Forest Regressor & Classifier**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes** 
*(Note: Classification approaches were explored by categorizing the RUL into discrete risk groups).*

### Deep Learning & Sequence Models
To capture the massive temporal and chronological significance of the time-series degradation:
- **LSTM (Long Short-Term Memory)**: Overcame vanilla RNN limits to learn long-term degradation dependencies.
- **Bidirectional LSTM (Bi-LSTM)**: Processed time-series telemetry in both forward and backward directions to vastly improve degradation pattern recognition.
- **CNN-Transformer Architectures**: A powerful hybrid approach leveraging Convolutional Neural Networks (CNN) for short-term spatial correlations and Transformer networks for un-vanishing long-term dependencies.

### Active Learning Strategies
An experimental notebook explores mitigating the lack of labeled data by proactively querying the most uncertain predictions.
- **Custom Uncertainty Sampling** (using squared error variance)
- **Monte Carlo (MC) Dropout** (estimating model uncertainty by maintaining dropout during inference)
- **Ensemble Methods**

## 📏 Model Evaluation Metrics
Given the high-stakes nature of aviation, evaluation went beyond standard regressions:
1. **RMSE (Root Mean Squared Error)** & **R² (R-Squared)**: Standard metrics to evaluate deviation.
2. **Asymmetric NASA S-Score**: A domain-specific penalizing metric. In aviation, **overestimating** an engine's life is drastically more dangerous than underestimating it. The S-Score heavily penalizes late predictions while being more generous to early maintenance calls.
3. **Hyperparameter Tuning**: Fully optimized using **KerasTuner** (RandomSearch applied to network depth, layers, and dropout rates).

## 📊 Explainability & Visualization
- **SmoothGrad (Grad x Input)**: Implemented within the CNN-Transformer notebook to attribute specific failure predictions back to individual subsystems. 
- **Feature Importance**: Mapped top contributing sensors via random forest evaluations.

## 📂 Repository Structure

```text
📁 RUL/
├── 📁 Mini Project 1/
│   ├── 📁 C-MPASS/                  # Raw and processed datasets
│   ├── 📄 requirements.txt          # Python dependencies
│   ├── 📄 bi-lstm_rul.ipynb         # Bidirectional LSTM model implementation
│   ├── 📄 engine-rul-cnn-transformer-w-failure-report.ipynb
│   ├── 📄 lstm-al-approach-to-predict-aircraft-engine-rul.ipynb
│   ├── 📄 cmapss-rul-predictivemaintenance-eda-modeling.ipynb
│   ├── 📄 predictive-maintenance-on-nasa-turbofan-jet-engine.ipynb
│   ├── 📄 Mini_Project_RUL.ipynb    # Core project notebook
│   └── 📄 ... (Various EDA notebooks, project reports, and presentations)
```

## 💻 Tech Stack
- **Languages**: Python
- **Core Libraries**: NumPy, Pandas, SciPy
- **Machine & Deep Learning**: TensorFlow/Keras, PyTorch, Scikit-learn, XGBoost
- **Optimization & Explainability**: KerasTuner, Optuna, SHAP
- **Data Visualization**: Matplotlib, Seaborn

## 🚀 Installation and Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd RUL
   ```

2. Navigate closely into the actual working directory:
   ```bash
   cd "Mini Project 1"
   ```

3. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # On macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Launch Jupyter Notebook to explore the code:
   ```bash
   jupyter notebook
   ```
