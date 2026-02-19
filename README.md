# 📡 LSTM-Based 3D Localization and Mobility Prediction for 5G Networks

> A deep learning–based approach for predicting future 3D user positions in noisy 5G-like environments using an LSTM model.

---

## 🌐 Overview

This project simulates and predicts **3D user mobility** in next-generation 5G networks. The pipeline:

- 🗺️ Simulates 3D mobility trajectories
- 🔊 Injects measurement noise
- 🔢 Builds time-series sequences
- 🧠 Trains an LSTM model
- 📏 Evaluates 3D positioning error

---

## 📂 Project Structure

```
project/
│
├── generate_dataset.py       # Simulates 3D trajectories with noise
├── preprocess_data.py        # Normalizes and sequences the data
├── train_lstm.py             # Trains the LSTM model
├── evaluate_model.py         # Evaluates model and plots results
├── kalman_baseline.py        # (Optional / Future Work)
├── 3d_mobility_dataset.csv   # Generated dataset
├── *.npy files               # Preprocessed train/val/test arrays
├── lstm_model.pth            # Saved trained model
└── README.md
```

> ⚠️ The `venv/` folder is intentionally **not included** in this repository.

---

## ⚙️ Requirements

- 🐍 **Python 3.11** *(Recommended)*
- 📦 **pip**

**Tested on:**
- 🪟 Windows 10 / 11
- 🐍 Python 3.11.x

---

## 🚀 Setup Instructions (After Cloning)

### 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

---

### 2️⃣ Create a Virtual Environment

> ⚠️ Make sure **Python 3.11** is installed before proceeding.

**Check available Python versions:**
```bash
py -0
```

**Create the virtual environment:**
```bash
py -3.11 -m venv venv
```

**Activate the environment:**

| Platform | Command |
|----------|---------|
| 🪟 Windows | `venv\Scripts\activate` |
| 🍎 Mac / 🐧 Linux | `source venv/bin/activate` |

**Verify your Python version:**
```bash
python --version
# Expected output: Python 3.11.x
```

---

### 3️⃣ Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch joblib
```

---

## ▶️ How to Run the Project (Step-by-Step)

Follow these steps **in order**:

---

### 🔹 Step 1 — Generate Dataset

```bash
python generate_dataset.py
```

**This will:**
- ✅ Create simulated 3D trajectories
- ✅ Inject Gaussian noise
- ✅ Save `3d_mobility_dataset.csv`

---

### 🔹 Step 2 — Preprocess Data

```bash
python preprocess_data.py
```

**This will:**
- ✅ Normalize the data
- ✅ Create time-window sequences
- ✅ Split into train / val / test sets
- ✅ Save `.npy` files
- ✅ Save scalers for inverse transformation

---

### 🔹 Step 3 — Train LSTM Model

```bash
python train_lstm.py
```

**This will:**
- ✅ Train the LSTM model
- ✅ Print training and validation loss per epoch
- ✅ Save the trained model as `lstm_model.pth`

---

### 🔹 Step 4 — Evaluate Model

```bash
python evaluate_model.py
```

**This will:**
- ✅ Load the trained model
- ✅ Predict test positions
- ✅ Compute RMSE for each axis
- ✅ Display a 3D trajectory comparison plot

---

## 📊 Expected Output

**Example evaluation metrics:**

```
RMSE X (m): ~25
RMSE Y (m): ~38
RMSE Z (m): ~19
Mean 3D Positioning Error: ~42 meters
```

> 📈 A **3D plot** comparing predicted vs. true trajectories will also be displayed.

---

## 🧠 Methodology Overview

| Component | Details |
|-----------|---------|
| 🔢 **Input** | 10 previous noisy 3D positions |
| 🏗️ **Model** | 2-layer LSTM (hidden size 64) |
| 🎯 **Output** | Next true 3D coordinate |
| 📉 **Loss Function** | Mean Squared Error (MSE) |
| ⚙️ **Optimizer** | Adam |



---

## 📝 Notes

- 🔬 This project uses **synthetic mobility data**.
- 🎓 It is a **research prototype** for academic purposes.
- 🐍 **Python 3.11** is recommended for stability with PyTorch.

---

## 📄 License

This project is intended for **academic and research use only**.
Owner : Shashank Jha - VIT Chennai | shashankshiv.jha@gmail.com
