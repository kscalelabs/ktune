import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# Read the CSV file
df = pd.read_csv('simulation_data.csv')

# Calculate projected gravity from quaternions
projected_gravity = []
for idx, row in df.iterrows():
    # Get quaternion components
    quat = [row['quat_w'], row['quat_x'], row['quat_y'], row['quat_z']]
    # Check if quaternion has non-zero norm
    quat_norm = np.linalg.norm(quat)
    
    if quat_norm > 1e-10:  # Small threshold to check for effectively zero norm
        # Create rotation from quaternion (note the order in scipy is [w, x, y, z])
        r = R.from_quat(quat, scalar_first=True)
        # Apply inverse rotation to [0, 0, 1] (gravity vector in world frame)
        proj_grav_world = r.apply([0.0, 0.0, 1.0], inverse=True)
    else:
        # If quaternion has zero norm, use NaN values
        print(f"Warning: Skipping gravity projection at row {idx}: quaternion has zero norm")
        proj_grav_world = [float('nan'), float('nan'), float('nan')]
    
    projected_gravity.append(proj_grav_world)

# Convert to array and add to dataframe
projected_gravity = np.array(projected_gravity)
df['proj_grav_x'] = projected_gravity[:, 0]
df['proj_grav_y'] = projected_gravity[:, 1]
df['proj_grav_z'] = projected_gravity[:, 2]

# Enhanced frequency analysis with duplicate detection
# Create a combined feature to detect any change in sensor readings
df['data_hash'] = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].astype(str).sum(axis=1)

# Detect duplicates and count them
df['is_duplicate'] = df['data_hash'].shift() == df['data_hash']
df.loc[0, 'is_duplicate'] = False  # First row is never a duplicate

# Count statistics
total_readings = len(df)
total_duplicates = df['is_duplicate'].sum()
duplicate_percent = (total_duplicates / total_readings) * 100 if total_readings > 0 else 0
runtime = df['sim_time'].max() - df['sim_time'].min()

# Find indices where data actually changes (non-duplicates)
data_change_indices = df.index[~df['is_duplicate']].tolist()

# Calculate frequencies
raw_hz = total_readings / runtime if runtime > 0 else 0
effective_hz = (total_readings - total_duplicates) / runtime if runtime > 0 else 0

# Calculate average time difference between unique readings
if len(data_change_indices) > 1:
    real_time_diff = np.diff([df.iloc[i]['sim_time'] for i in data_change_indices])
    avg_time_diff = np.mean(real_time_diff)
    unique_update_hz = 1 / avg_time_diff if avg_time_diff > 0 else 0
else:
    unique_update_hz = 0

# Print frequency analysis summary
print(f"Stats: {total_duplicates} duplicates/{total_readings} total readings ({duplicate_percent:.1f}%) over {runtime:.1f}s")
print(f"Effective rate: {effective_hz:.1f} Hz (Raw: {raw_hz:.1f} Hz)")
print(f"Average frequency between unique updates: {unique_update_hz:.2f} Hz")

# Create a figure with 5 subplots (3x2 grid with the last position empty)
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])

title_str = input("Enter file name: ")
fig.suptitle(f'IMU Analysis {title_str} - Raw: {raw_hz:.1f} Hz, Effective: {effective_hz:.1f} Hz, Duplicates: {duplicate_percent:.1f}%', fontsize=16)

# Plot accelerometer data
ax1.plot(df['sim_time'], df['acc_x'], label='X')
ax1.plot(df['sim_time'], df['acc_y'], label='Y')
ax1.plot(df['sim_time'], df['acc_z'], label='Z')
ax1.set_title('Accelerometer Data')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (m/s²)')
ax1.legend()
ax1.grid(True)

# Plot gyroscope data with ±2 rad/s threshold
ax2.plot(df['sim_time'], df['gyro_x'], label='X')
ax2.plot(df['sim_time'], df['gyro_y'], label='Y')
ax2.plot(df['sim_time'], df['gyro_z'], label='Z')
# ax2.axhline(y=2, color='r', linestyle='--', alpha=0.5)
# ax2.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
# ax2.fill_between(df['sim_time'], -2, 2, color='gray', alpha=0.2)
ax2.set_title('Gyroscope Data')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.legend()
ax2.grid(True)

# Plot projected gravity
ax3.plot(df['sim_time'], df['proj_grav_x'], label='X')
ax3.plot(df['sim_time'], df['proj_grav_y'], label='Y')
ax3.plot(df['sim_time'], df['proj_grav_z'], label='Z')
ax3.set_title('Projected Gravity')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Gravity Component')
ax3.legend()
ax3.grid(True)

# Plot quaternion data
ax4.plot(df['sim_time'], df['quat_x'], label='X')
ax4.plot(df['sim_time'], df['quat_y'], label='Y')
ax4.plot(df['sim_time'], df['quat_z'], label='Z')
ax4.plot(df['sim_time'], df['quat_w'], label='W')
ax4.set_title('Quaternion')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Value')
ax4.legend()
ax4.grid(True)

# Plot base_link angles directly from the CSV
ax5.plot(df['sim_time'], df['base_link_x'], label='X')
ax5.plot(df['sim_time'], df['base_link_y'], label='Y')
ax5.plot(df['sim_time'], df['base_link_z'], label='Z')
ax5.set_title('Base Link Angles')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Angle (degrees)')
ax5.legend()
ax5.grid(True)

# Calculate average quaternion values
avg_quat_x = df['quat_x'].mean()
avg_quat_y = df['quat_y'].mean()
avg_quat_z = df['quat_z'].mean()
avg_quat_w = df['quat_w'].mean()

# Print average quaternion values
print(f"Average quaternion values:")
print(f"  X: {avg_quat_x:.6f}")
print(f"  Y: {avg_quat_y:.6f}")
print(f"  Z: {avg_quat_z:.6f}")
print(f"  W: {avg_quat_w:.6f}")

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(f'{title_str}_imu_plots.png')
# plt.show() 