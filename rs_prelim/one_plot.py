import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json
import glob
import argparse
import numpy as np

def load_data_from_file(file_path):
    """Load data from a JSON file and return a DataFrame with the data."""
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    # Extract metadata and data points
    meta_data = json_data['config']
    data_points = json_data['data']
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    return df, meta_data

def load_metadata():
    """Load actuator metadata from metadata.json."""
    with open('metadata.json', 'r') as f:
        return json.load(f)

def plot_actuator_subplots(actuator_files, data_type, output_file):
    """Create a vertical stack of sub-plots, one per actuator."""
    # Keep only actuator IDs that have BOTH real & sim files
    valid_ids = [aid for aid, f in actuator_files.items()
                 if f[f'{data_type}_real'] and f[f'{data_type}_sim']]
    if not valid_ids:
        print(f"No valid actuator {data_type} data found.")
        return

    # ------------------------------------------------------------------
    n = len(valid_ids)
    fig, axes = plt.subplots(n, 1, figsize=(18, 4 * n), sharex=True)

    if n == 1:                       # make iterable if only one subplot
        axes = [axes]
    # ------------------------------------------------------------------

    for ax, actuator_id in zip(axes, sorted(valid_ids)):
        files = actuator_files[actuator_id]
        real_df, real_meta = load_data_from_file(files[f'{data_type}_real'])
        sim_df, sim_meta = load_data_from_file(files[f'{data_type}_sim'])

        ax.plot(real_df["time_since_start"], real_df["commanded_position"],
                "k-",  linewidth=2,  label="Commanded")
        ax.plot(real_df["time_since_start"], real_df["position"],
                "r-", linewidth=1.2, label="Real")
        ax.plot(sim_df["time_since_start"],  sim_df["position"],
                "b--", linewidth=1.2, label="Sim")

        ax.set_ylabel("Pos (deg)", fontsize=11)
        
        # Updated title with armature and frictionloss from sim model
        ax.set_title(f"Actuator {actuator_id} - Kp={real_meta['kp']}, Kd={real_meta['kd']}, MaxTq={real_meta['max_torque']}\n"
                     f"Sim params: Armature={sim_meta['armature']}, Friction={sim_meta['frictionloss']}, Damping={sim_meta['damping']}, Actuatorfrcrange={sim_meta['actuatorfrcrange']}",
                     fontsize=12)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    response_type = data_type.capitalize()
    fig.suptitle(f"Real vs Simulation – {response_type} Response", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(output_file)
    print(f"Plot saved -> {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Overlay real & sim data for each actuator")
    parser.add_argument("--date", default="20250503",
                        help="Folder date (YYYYMMDD)")
    parser.add_argument("--right", action="store_true",
                        help="Process only right leg actuators (41-45)")
    args = parser.parse_args()

    # Build path INSIDE rs_prelim
    date_dir = os.path.join(args.date)
    if not os.path.isdir(date_dir):
        print(f"Directory '{date_dir}' not found.")
        sys.exit(1)

    # Define data type directories and their mapping
    data_dirs = {
        "step_real": "step_real",
        "step_sim": "step",
        "chirp_real": "chirp_real",
        "chirp_sim": "chirp"
    }
    
    # Verify directories exist
    for key, subdir in data_dirs.items():
        path = os.path.join(date_dir, subdir)
        if not os.path.isdir(path):
            print(f"Warning: Directory '{path}' not found.")
            data_dirs[key] = None

    # Load files from each directory
    actuator_files = {}
    
    # Process each data type directory
    for data_key, subdir in data_dirs.items():
        if not subdir:
            continue
            
        path = os.path.join(date_dir, subdir)
        files = glob.glob(os.path.join(path, "*.json"))
        
        if not files:
            print(f"No JSON files in '{path}'.")
            continue
            
        # Parse data type and source
        data_type, source = data_key.split("_")
        
        # Process files
        for fp in files:
            filename = os.path.basename(fp)
            parts = filename.split("_")
            
            # Files are named like: real_11_damp0.01.json or sim_11_damp0.01.json
            if len(parts) >= 2:
                act_id = parts[1]  # Extract actuator ID
                
                # Initialize actuator entry if needed
                actuator_files.setdefault(act_id, {})
                
                # Store file path with appropriate key
                key = f"{data_type}_{source}"
                actuator_files[act_id][key] = fp

    # Load metadata
    metadata = load_metadata()
    
    # Define actuator ID groups
    id_groups = {
        "arm_left": range(11, 16),  # 11-15
        "arm_right": range(21, 26), # 21-25
        "leg_left": range(31, 36),  # 31-35
        "leg_right": range(41, 46)  # 41-45
    }

    # Filter to only right leg if specified
    if args.right:
        id_groups = {"leg_right": id_groups["leg_right"]}

    # Create plots for each group and data type
    for group_name, ids in id_groups.items():
        group_actuators = {}
        for act_id in ids:
            act_id_str = str(act_id)
            if act_id_str in actuator_files:
                group_actuators[act_id_str] = actuator_files[act_id_str]
        
        if group_actuators:
            # Create step response plots
            plot_actuator_subplots(
                group_actuators, 
                'step', 
                f"{group_name}_step_response.png"
            )
            
            # Create chirp response plots
            plot_actuator_subplots(
                group_actuators, 
                'chirp', 
                f"{group_name}_chirp_response.png"
            )

if __name__ == "__main__":
    main()
