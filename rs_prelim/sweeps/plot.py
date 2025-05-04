import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
import datetime
import json
import glob
import argparse
def plot_actuator_data(file_path, title):
    # Read the JSON file
    with open(file_path, 'r') as f:
        json_data = json.load(f)

    # Assuming filename format like: sim_42_20250430_144743.json
    base_filename = os.path.basename(file_path)
    org_file_name = base_filename.split('.')[0]  # Remove extension
    
    meta_data = json_data['config']

    # Extract data points
    data_points = json_data['data']
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Create a figure with a specific size
    plt.figure(figsize=(12, 6))
    
    # Plot the position over time
    plt.plot(df['time_since_start'], df['position'], '-', linewidth=1)
    plt.plot(df['time_since_start'], df['commanded_position'], '--', linewidth=1)
    
    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Position (deg)')
    if meta_data['mode'] == 'sim':
        plt.title(f'{org_file_name} {title} \n Actuator {meta_data["actuator_id"]} Position over Time, start at {meta_data["start_pos"]}°\n'
              f'{meta_data["armature"]} armature, {meta_data["frictionloss"]} friction loss, {meta_data["actuatorfrcrange"]} actuatorfrcrange\n'
              f'{meta_data["chirp_duration"]}s chirp, {meta_data["start_freq"]}Hz start freq, {meta_data["end_freq"]}Hz end freq \n'
              f'{meta_data["kp"]} kp, {meta_data["kd"]} kd, {meta_data["max_torque"]} max torque')
    else:
         plt.title(f'{org_file_name} {title} \n Actuator {meta_data["actuator_id"]} Position over Time, start at {meta_data["start_pos"]}°\n'
              f'{meta_data["chirp_duration"]}s step, {meta_data["step_size"]}° step size \n'
              f'{meta_data["step_count"]} steps, \n{meta_data["kp"]} kp, {meta_data["kd"]} kd, {meta_data["max_torque"]} max torque')

   
    # Add secondary y-axis for torque if it exists
    if 'output_torque' in df.columns:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(df['time_since_start'], df['output_torque'], 'r-', linewidth=1)
        ax2.set_ylabel('Torque (Nm)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        plt.legend(['position', 'commanded position', 'output torque'], loc='upper right')
    else:
        plt.legend(['position', 'commanded position'])


    plt.legend(['position', 'commanded position'])
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    

    
    plt.savefig(f'{org_file_name}.png')

if __name__ == "__main__":
    # If file path is provided as an argument, use it
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=datetime.datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    # Look for *_resp_step_* JSON files
    files = glob.glob(f'{args.date}/*.json')

    
    if not files:
        # Fall back to other JSON files if none found
        files = glob.glob('*.json')
        
    if files:
        # Sort files by timestamp in the filename
        files.sort(key=lambda x: x.split('_')[-1].split('.')[0], reverse=True)
        file_path = files[0]
        print(f"Using newest file: {file_path}")
    else:
        print("No JSON data files found.")
        sys.exit(1)
    
    print(file_path)
    plot_actuator_data(file_path, args.title)
