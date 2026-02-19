import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use("TkAgg")   # Ensures plot window opens on Windows
import matplotlib.pyplot as plt


# ----- LSTM MODEL DEFINITION -----
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out


# ----- LOAD TEST DATA -----
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

X_test = torch.tensor(X_test, dtype=torch.float32)


# ----- LOAD MODEL -----
model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()


# ----- MAKE PREDICTIONS -----
with torch.no_grad():
    predictions = model(X_test)

predictions = predictions.numpy()


# ----- LOAD SCALER -----
target_scaler = joblib.load("target_scaler.save")


# ----- CONVERT BACK TO REAL METERS -----
pred_real = target_scaler.inverse_transform(predictions)
true_real = target_scaler.inverse_transform(y_test)


# ----- DEBUG PRINT -----
print("Sample True:", true_real[0])
print("Sample Pred:", pred_real[0])


# ----- PER-AXIS RMSE -----
rmse_x = np.sqrt(np.mean((pred_real[:,0] - true_real[:,0])**2))
rmse_y = np.sqrt(np.mean((pred_real[:,1] - true_real[:,1])**2))
rmse_z = np.sqrt(np.mean((pred_real[:,2] - true_real[:,2])**2))

print("RMSE X (m):", rmse_x)
print("RMSE Y (m):", rmse_y)
print("RMSE Z (m):", rmse_z)


# ----- 3D POSITIONING ERROR -----
distance_error = np.sqrt(
    (pred_real[:,0] - true_real[:,0])**2 +
    (pred_real[:,1] - true_real[:,1])**2 +
    (pred_real[:,2] - true_real[:,2])**2
)

print("Mean 3D Positioning Error (meters):", np.mean(distance_error))


# ----- 3D TRAJECTORY VISUALIZATION -----
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(true_real[:200,0], true_real[:200,1], true_real[:200,2], label="True")
ax.plot(pred_real[:200,0], pred_real[:200,1], pred_real[:200,2], label="Predicted")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()
