import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

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

model = LSTMModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    
    model.train()
    
    permutation = torch.randperm(X_train.size()[0])
    
    epoch_loss = 0
    
    for i in range(0, X_train.size()[0], BATCH_SIZE):
        
        indices = permutation[i:i+BATCH_SIZE]
        batch_x = X_train[indices]
        batch_y = y_train[indices]
        
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Train Loss: {epoch_loss:.4f}, "
          f"Val Loss: {val_loss.item():.4f}")

torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved successfully.")
