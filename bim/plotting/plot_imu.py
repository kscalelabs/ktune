import time
import argparse
import matplotlib.pyplot as plt
import sys
import os
import imu

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
        "-d", "--duration", type=float, default=2.0, help="Duration to read IMU data in seconds (default: 2.0)"
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


    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Create 2 subplots, sharing the x-axis

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
        # axs[1].set_ylim(bottom=0) # Optional: Set y-axis lower limit if needed
        axs[1].grid(True)
        axs[1].legend()
    else:
        axs[1].set_title('Sampling Frequency (Not enough data)')
        axs[1].grid(True)


    axs[1].set_xlabel('Time (s)') # X-axis label only needed for the bottom plot due to sharex=True
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels

    # print("Displaying plot...")
    # plt.show()
    output_filename = "imu_quaternion_plot.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    # print("Plot window closed.")

if __name__ == "__main__":
    main()
