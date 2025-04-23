import time
import argparse
import matplotlib.pyplot as plt
import sys
import os
import imu
import numpy as np
import math
from scipy.spatial.transform import Rotation

# Try printing the module representation and its path
print(f"Imported imu module: {imu}")
try:
    print(f"Imported imu path: {imu.__path__}")
except AttributeError:
    print("Imported imu module has no __path__ attribute.")

# The original line that caused the TypeError
# print(f"IMU module path: {os.path.abspath(imu.__file__)}")

# # Add the parent directory of 'ktune' to the Python path
# # This allows importing the 'imu' module from 'ktune/bim/imu'
# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir) # This should be ktune/bim
# grandparent_dir = os.path.dirname(parent_dir) # This should be ktune
# greatgrandparent_dir = os.path.dirname(grandparent_dir) # This should be the workspace root
# sys.path.insert(0, greatgrandparent_dir)




def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Read and plot quaternion data from Hiwonder IMU.")
    parser.add_argument(
        "-d", "--duration", type=float, default=5.0, help="Duration to read IMU data in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--device", type=str, default="/dev/ttyUSB0", help="Serial device path for the IMU (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate", type=int, default=230400, help="Serial baud rate for the IMU (default: 230400)"
    )
    return parser.parse_args()

def main():
    """Main function to read IMU data and plot quaternions."""
    args = parse_args()
    read_duration = args.duration

    print(f"Attempting to connect to IMU at {args.device} with baudrate {args.baudrate}...")
    try:
        reader = imu.create_hiwonder(args.device, args.baudrate)
        print("IMU connection successful.")
    except Exception as e:
        print(f"Error connecting to IMU: {e}")
        print("Please check the device path, permissions, and baudrate.")
        return

    timestamps = []
    quat_w = []
    quat_x = []
    quat_y = []
    quat_z = []
    
    # Lists to store projected gravity values
    proj_grav_x = []
    proj_grav_y = []
    proj_grav_z = []
    
    # Lists to store Euler angles directly from IMU
    imu_roll = []
    imu_pitch = []
    imu_yaw = []
    
    # Lists to store gyroscope data
    gyro_x = []
    gyro_y = []
    gyro_z = []
    
    # Lists to store accelerometer data
    accel_x = []
    accel_y = []
    accel_z = []


    print(f"Reading IMU data for {read_duration:.1f} seconds...")
    start_time = time.monotonic()
    try:
        while (current_time := time.monotonic()) - start_time < read_duration:
            if data := reader.get_data():
                if 'quaternion' in data and data['quaternion'] is not None:
                    q = data['quaternion'] # Assuming order is (w, x, y, z) based on example
                    elapsed_time = current_time - start_time

                    timestamps.append(elapsed_time)
                    # Access quaternion components using attributes
                    quat_w.append(q.w)
                    quat_x.append(q.x)
                    quat_y.append(q.y)
                    quat_z.append(q.z)
                    
                    # Calculate projected gravity vector
                    r = Rotation.from_quat([q.w, q.x, q.y, q.z], scalar_first=True)
                    proj_grav_world = r.apply(np.array([0.0, 0.0, 1.0]), inverse=True)
                    projected_gravity = proj_grav_world
                    proj_grav_x.append(projected_gravity[0])
                    proj_grav_y.append(projected_gravity[1])
                    proj_grav_z.append(projected_gravity[2])
                    
                # Get Euler angles directly from IMU
                if 'euler' in data and data['euler'] is not None:
                    euler = data['euler']
                    imu_roll.append(euler.x * 180 / math.pi)  # Assuming euler has x, y, z attributes for roll, pitch, yaw
                    imu_pitch.append(euler.y * 180 / math.pi)
                    imu_yaw.append(euler.z * 180 / math.pi)
                
                # Get gyroscope data
                if 'gyroscope' in data and data['gyroscope'] is not None:
                    gyro = data['gyroscope']
                    gyro_x.append(gyro.x)  # Assuming gyroscope has x, y, z attributes
                    gyro_y.append(gyro.y)
                    gyro_z.append(gyro.z)
                    
                # Get accelerometer data
                if 'accelerometer' in data and data['accelerometer'] is not None:
                    accel = data['accelerometer']
                    accel_x.append(accel.x)  # Assuming accelerometer has x, y, z attributes
                    accel_y.append(accel.y)
                    accel_z.append(accel.z)

            time.sleep(0.005) # Small delay to prevent high CPU usage, adjust if needed

    except KeyboardInterrupt:
        print("Stopping data collection early.")
    finally:
        # Potentially add reader cleanup if the library requires it, e.g., reader.close()
        print("Finished reading data.")
        pass # No explicit close method observed in example

    if not timestamps:
        print("No data collected. Cannot generate plot.")
        return

    print(f"Collected {len(timestamps)} data points.")

    # Calculate sampling frequencies
    frequencies = []
    frequency_timestamps = []
    if len(timestamps) > 1:
        for i in range(1, len(timestamps)):
            delta_t = timestamps[i] - timestamps[i-1]
            if delta_t > 1e-9: # Avoid division by zero or near-zero
                freq = 1.0 / delta_t
                frequencies.append(freq)
                frequency_timestamps.append(timestamps[i]) # Time corresponding to the frequency calculation
            # else:
                # Optionally handle cases with zero delta_t if needed
                # print(f"Warning: Zero or very small time difference detected at index {i}")
        print(f"Calculated {len(frequencies)} frequency points.")
    else:
        print("Not enough data points to calculate frequency.")

    # Calculate cumulative average frequencies
    average_frequencies = []
    average_freq_timestamps = []
    if len(timestamps) > 1:
        for i in range(1, len(timestamps)):
             # Average frequency = (total samples) / (total time)
             # At timestamps[i], we have received i+1 samples
             # Total time elapsed is timestamps[i]
            if timestamps[i] > 1e-9: # Avoid division by zero
                avg_freq = (i + 1) / timestamps[i]
                average_frequencies.append(avg_freq)
                average_freq_timestamps.append(timestamps[i])
        print(f"Calculated {len(average_frequencies)} average frequency points.")


    # Plotting
    fig, axs = plt.subplots(8, 1, figsize=(12, 28), sharex=True) # Create 8 subplots, sharing the x-axis

    # Plot Quaternion Data on the first subplot
    axs[0].plot(timestamps, quat_w, label='W')
    axs[0].plot(timestamps, quat_x, label='X')
    axs[0].plot(timestamps, quat_y, label='Y')
    axs[0].plot(timestamps, quat_z, label='Z')
    axs[0].set_title(f'IMU Quaternion Data ({read_duration:.1f}s)')
    axs[0].set_ylabel('Quaternion Component Value')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Frequency Data on the second subplot
    if frequency_timestamps:
        axs[1].plot(frequency_timestamps, frequencies, label='Frequency', marker='.', linestyle='-')
        axs[1].set_title('Sampling Frequency')
        axs[1].set_ylabel('Frequency (Hz)')
        axs[1].grid(True)
        axs[1].legend()
    else:
        axs[1].set_title('Sampling Frequency (Not enough data)')
        axs[1].grid(True)

    # Plot Average Frequency Data on the third subplot
    if average_freq_timestamps:
        axs[2].plot(average_freq_timestamps, average_frequencies, label='Average Frequency', linestyle='-')
        axs[2].set_title('Cumulative Average Sampling Frequency')
        axs[2].set_ylabel('Average Frequency (Hz)')
        axs[2].grid(True)
        axs[2].legend()
    else:
        axs[2].set_title('Cumulative Average Sampling Frequency (Not enough data)')
        axs[2].grid(True)

    # Plot Projected Gravity Data on the fourth subplot
    axs[3].plot(timestamps, proj_grav_x, label='X', linestyle='-')
    axs[3].plot(timestamps, proj_grav_y, label='Y', linestyle='-')
    axs[3].plot(timestamps, proj_grav_z, label='Z', linestyle='-')
    axs[3].set_title('Projected Gravity Vectors')
    axs[3].set_ylabel('Gravity Vector Component Value')
    axs[3].legend()
    axs[3].grid(True)

    # Calculate Euler angles from quaternions
    roll = []
    pitch = []
    yaw = []
    
    for w, x, y, z in zip(quat_w, quat_x, quat_y, quat_z):
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll_angle = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch_angle = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw_angle = math.atan2(siny_cosp, cosy_cosp)

        # Convert to degrees
        roll.append(math.degrees(roll_angle))
        pitch.append(math.degrees(pitch_angle))
        yaw.append(math.degrees(yaw_angle))
        
    # Plot Calculated Euler Angles on the fifth subplot
    axs[4].plot(timestamps, roll, label='Roll', linestyle='-')
    axs[4].plot(timestamps, pitch, label='Pitch', linestyle='-')
    axs[4].plot(timestamps, yaw, label='Yaw', linestyle='-')
    axs[4].set_title('Calculated Euler Angles from Quaternions')
    axs[4].set_ylabel('Angle (degrees)')
    axs[4].legend()
    axs[4].grid(True)
    
    # Plot IMU Euler Angles on the sixth subplot
    if imu_roll and imu_pitch and imu_yaw:  # Check if we have Euler data
        axs[5].plot(timestamps[:len(imu_roll)], imu_roll, label='Roll', linestyle='-')
        axs[5].plot(timestamps[:len(imu_pitch)], imu_pitch, label='Pitch', linestyle='-')
        axs[5].plot(timestamps[:len(imu_yaw)], imu_yaw, label='Yaw', linestyle='-')
        axs[5].set_title('IMU Euler Angles (Direct Measurements)')
        axs[5].set_ylabel('Angle (Degrees)')
        axs[5].legend()
        axs[5].grid(True)
    else:
        axs[5].set_title('IMU Euler Angles')
        axs[5].grid(True)
    
    # Plot Gyroscope Data on the seventh subplot
    if gyro_x and gyro_y and gyro_z:  # Check if we have gyroscope data
        axs[6].plot(timestamps[:len(gyro_x)], gyro_x, label='X', linestyle='-')
        axs[6].plot(timestamps[:len(gyro_y)], gyro_y, label='Y', linestyle='-')
        axs[6].plot(timestamps[:len(gyro_z)], gyro_z, label='Z', linestyle='-')
        axs[6].set_title('Gyroscope Data')
        axs[6].set_ylabel('Angular Velocity (deg/s)')
        axs[6].legend()
        axs[6].grid(True)
    else:
        axs[6].set_title('Gyroscope Data')
        axs[6].grid(True)
        
    # Plot Accelerometer Data on the eighth subplot
    if accel_x and accel_y and accel_z:  # Check if we have accelerometer data
        axs[7].plot(timestamps[:len(accel_x)], accel_x, label='X', linestyle='-')
        axs[7].plot(timestamps[:len(accel_y)], accel_y, label='Y', linestyle='-')
        axs[7].plot(timestamps[:len(accel_z)], accel_z, label='Z', linestyle='-')
        axs[7].set_title('Accelerometer Data')
        axs[7].set_ylabel('Acceleration (m/s²)')
        axs[7].legend()
        axs[7].grid(True)
    else:
        axs[7].set_title('Accelerometer Data')
        axs[7].grid(True)

    axs[7].set_xlabel('Time (s)') # X-axis label only needed for the bottom plot due to sharex=True
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels

    # print("Displaying plot...")
    output_filename = "imu_quaternion_plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    # print("Plot window closed.")

if __name__ == "__main__":
    main()
