import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib


# ----- LOAD DATA -----
df = pd.read_csv("3d_mobility_dataset.csv")

print("Dataset loaded successfully")
print(df.head())


# ----- DEFINE SCALERS (FIT ON FULL DATA) -----
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(df[["x_noisy", "y_noisy", "z_noisy"]].values)
target_scaler.fit(df[["x", "y", "z"]].values)


# ----- CREATE SEQUENCES PER MOTION TYPE -----
TIME_STEPS = 10

X_all = []
y_all = []

for motion in df["motion_type"].unique():
    
    df_motion = df[df["motion_type"] == motion].reset_index(drop=True)
    
    features_motion = df_motion[["x_noisy", "y_noisy", "z_noisy"]].values
    targets_motion = df_motion[["x", "y", "z"]].values
    
    features_scaled = feature_scaler.transform(features_motion)
    targets_scaled = target_scaler.transform(targets_motion)
    
    for i in range(len(features_scaled) - TIME_STEPS):
        X_all.append(features_scaled[i:i+TIME_STEPS])
        y_all.append(targets_scaled[i+TIME_STEPS])


X = np.array(X_all)
y = np.array(y_all)

print("Sequence shape:", X.shape)
print("Target shape:", y.shape)


# ----- TRAIN / VALIDATION / TEST SPLIT -----
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    shuffle=True
)

print("X_train shape:", X_train.shape)
print("X_temp shape:", X_temp.shape)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    shuffle=True
)

print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)


# ----- SAVE DATA -----
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("All processed datasets saved successfully.")


# ----- SAVE SCALERS -----
joblib.dump(feature_scaler, "feature_scaler.save")
joblib.dump(target_scaler, "target_scaler.save")

print("Scalers saved successfully.")
