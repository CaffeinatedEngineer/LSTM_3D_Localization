import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

TOTAL_STEPS = 1000
TIME = np.arange(TOTAL_STEPS)

def generate_linear_motion():
    vx, vy, vz = 0.5, 0.3, 0.2
    
    x = vx * TIME
    y = vy * TIME
    z = vz * TIME
    
    return x, y, z

def generate_circular_motion():
    radius = 50
    omega = 0.02
    
    x = radius * np.cos(omega * TIME)
    y = radius * np.sin(omega * TIME)
    z = 0.1 * TIME
    
    return x, y, z

def generate_random_walk():
    x = np.cumsum(np.random.normal(0, 1, TOTAL_STEPS))
    y = np.cumsum(np.random.normal(0, 1, TOTAL_STEPS))
    z = np.cumsum(np.random.normal(0, 0.5, TOTAL_STEPS))
    
    return x, y, z

def generate_accelerated_motion():
    ax, ay, az = 0.01, 0.015, 0.005
    
    x = 0.5 * ax * TIME**2
    y = 0.5 * ay * TIME**2
    z = 0.5 * az * TIME**2
    
    return x, y, z

def add_noise(x, y, z):
    noise_level = 5  # meters
    
    x_noisy = x + np.random.normal(0, noise_level, len(x))
    y_noisy = y + np.random.normal(0, noise_level, len(y))
    z_noisy = z + np.random.normal(0, noise_level, len(z))
    
    return x_noisy, y_noisy, z_noisy

motions = {
    "linear": generate_linear_motion,
    "circular": generate_circular_motion,
    "random": generate_random_walk,
    "accelerated": generate_accelerated_motion
}

all_data = []

for motion_name, motion_func in motions.items():
    
    x, y, z = motion_func()
    x_noisy, y_noisy, z_noisy = add_noise(x, y, z)
    
    df = pd.DataFrame({
        "time": TIME,
        "motion_type": motion_name,
        "x": x,
        "y": y,
        "z": z,
        "x_noisy": x_noisy,
        "y_noisy": y_noisy,
        "z_noisy": z_noisy
    })
    
    all_data.append(df)

final_dataset = pd.concat(all_data, ignore_index=True)


final_dataset.to_csv("3d_mobility_dataset.csv", index=False)

print("Dataset generated successfully!")
print(final_dataset.head())

