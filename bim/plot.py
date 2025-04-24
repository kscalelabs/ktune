import pickle
import os
from IMUData import IMUData

# Path to the pickle file
pickle_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/20250424_real/amp50.0_freq0.5_010657.pkl')

# Load the pickle file
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
    # Available keys in the data dictionary:
    # 'amplitude', 'frequency', 'timestamp', 'loop_freq', 
    # 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
    # 'quat_w', 'quat_x', 'quat_y', 'quat_z', 
    # 'euler_x_quat', 'euler_y_quat', 'euler_z_quat',
    # 'euler_x', 'euler_y', 'euler_z',
    # 'proj_x', 'proj_y', 'proj_z'

# Print the contents of the data object
print("Data object contents:")
print(f"Amplitude: {data['amplitude']}")
print(f"Frequency: {data['frequency']}")
print(f"Number of samples: {len(data['timestamp'])}")
