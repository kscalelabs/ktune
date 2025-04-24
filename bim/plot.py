import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import colorlogging
import logging
import glob
from datetime import datetime

logger = logging.getLogger(__name__)
colorlogging.configure()


"""
# Available keys in the data dictionary:
# 'amplitude', 'frequency', 'timestamp', 'loop_freq', 
# 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
# 'quat_w', 'quat_x', 'quat_y', 'quat_z', 
# 'euler_x_quat', 'euler_y_quat', 'euler_z_quat',
# 'euler_x', 'euler_y', 'euler_z',
# 'proj_x', 'proj_y', 'proj_z'
"""



def plot_side_by_side(sim, real, plot_title, filetitle):
    """Figure 1: side-by-side subplots"""
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 15), constrained_layout=True)
    fig.suptitle(plot_title, fontsize=13, fontweight='bold')
    t_sim = sim['timestamp']
    t_real = real['timestamp']

    # 1) quaternions (w,x,y,z)
    sim_quats = sim['quat_w'] + sim['quat_x'] + sim['quat_y'] + sim['quat_z']
    real_quats = real['quat_w'] + real['quat_x'] + real['quat_y'] + real['quat_z']
    qmin, qmax = min(sim_quats + real_quats)*1.1, max(sim_quats + real_quats)*1.1

    for ax, data, title in [
        (axes[0,0], sim, 'Simulated Quaternions'),
        (axes[0,1], real, 'Real Quaternions')
    ]:
        ax.plot(t_sim if data is sim else t_real, data['quat_w'], marker='.', markersize=4, markevery=8, label='w')
        ax.plot(t_sim if data is sim else t_real, data['quat_x'], marker='.', markersize=4, markevery=10, label='x')
        ax.plot(t_sim if data is sim else t_real, data['quat_y'], marker='.', markersize=4, markevery=12, label='y')
        ax.plot(t_sim if data is sim else t_real, data['quat_z'], marker='.', markersize=4, markevery=14, label='z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Quaternion Value', fontsize=10)
        ax.set_ylim(qmin, qmax)
        ax.grid(True)
        ax.legend()

    # 2) euler angles from quat
    sim_e = sim['euler_x_quat'] + sim['euler_y_quat'] + sim['euler_z_quat']
    real_e = real['euler_x_quat'] + real['euler_y_quat'] + real['euler_z_quat']
    emin, emax = min(sim_e + real_e)*1.1, max(sim_e + real_e)*1.1

    for ax, data, title in [
        (axes[1,0], sim, 'Simulated Euler (from quat)'),
        (axes[1,1], real, 'Real Euler (from quat)')
    ]:
        ax.plot(t_sim if data is sim else t_real, data['euler_x_quat'], label='roll')
        ax.plot(t_sim if data is sim else t_real, data['euler_y_quat'], label='pitch')
        ax.plot(t_sim if data is sim else t_real, data['euler_z_quat'], label='yaw')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Degrees', fontsize=10)
        ax.set_ylim(emin, emax)
        ax.grid(True)
        ax.legend()

    # 3) gyroscope data
    sim_g = sim['gyro_x'] + sim['gyro_y'] + sim['gyro_z']
    real_g = real['gyro_x'] + real['gyro_y'] + real['gyro_z']
    gmin, gmax = min(sim_g + real_g)*1.1, max(sim_g + real_g)*1.1

    for ax, data, title in [
        (axes[2,0], sim, 'Simulated Gyroscope'),
        (axes[2,1], real, 'Real Gyroscope')
    ]:
        ax.plot(t_sim if data is sim else t_real, data['gyro_x'], label='gyro_x')
        ax.plot(t_sim if data is sim else t_real, data['gyro_y'], label='gyro_y')
        ax.plot(t_sim if data is sim else t_real, data['gyro_z'], label='gyro_z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Rad/s', fontsize=10)
        ax.set_ylim(gmin, gmax)
        ax.grid(True)
        ax.legend()

    # 4) accelerometer data
    sim_a = sim['acc_x'] + sim['acc_y'] + sim['acc_z']
    real_a = real['acc_x'] + real['acc_y'] + real['acc_z']
    amin, amax = min(sim_a + real_a)*1.1, max(sim_a + real_a)*1.1

    for ax, data, title in [
        (axes[3,0], sim, 'Simulated Accelerometer'),
        (axes[3,1], real, 'Real Accelerometer')
    ]:
        ax.plot(t_sim if data is sim else t_real, data['acc_x'], label='acc_x')
        ax.plot(t_sim if data is sim else t_real, data['acc_y'], label='acc_y')
        ax.plot(t_sim if data is sim else t_real, data['acc_z'], label='acc_z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('m/s²', fontsize=10)
        ax.set_ylim(amin, amax)
        ax.grid(True)
        ax.legend()

    # 5) projected gravity
    sim_p = sim['proj_x'] + sim['proj_y'] + sim['proj_z']
    real_p = real['proj_x'] + real['proj_y'] + real['proj_z']
    pmin, pmax = min(sim_p + real_p)*1.1, max(sim_p + real_p)*1.1

    for ax, data, title in [
        (axes[4,0], sim, 'Simulated Projected'),
        (axes[4,1], real, 'Real Projected')
    ]:
        ax.plot(t_sim if data is sim else t_real, data['proj_x'], label='proj_x')
        ax.plot(t_sim if data is sim else t_real, data['proj_y'], label='proj_y')
        ax.plot(t_sim if data is sim else t_real, data['proj_z'], label='proj_z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Magnitude', fontsize=10)
        ax.set_ylim(pmin, pmax)
        ax.grid(True)
        ax.legend()
        
    # Save the figure
    # Get the timestamp from the filename stored in the data dictionary
    output_dir = os.path.dirname(real.get('_filename', ''))
    if output_dir:
        save_path = os.path.join(output_dir, f'side_by_side_{filetitle}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved side-by-side plot to {save_path}")
    
    return fig

def plot_overlay(sim, real, plot_title, filetitle):
    """Figure 2: overlay sim vs real"""
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 16), constrained_layout=True)
    fig.suptitle(plot_title, fontsize=13, fontweight='bold')
    t_sim, t_real = sim['timestamp'], real['timestamp']

    def common_limits(ax, sim_vals, real_vals):
        vmin = min(min(sim_vals), min(real_vals))
        vmax = max(max(sim_vals), max(real_vals))
        ax.set_ylim(vmin, vmax)
        ax.set_xlim(min(t_sim[0], t_real[0])*1.1, max(t_sim[-1], t_real[-1])*1.1)
        ax.grid(True)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 1) Quaternions overlay
    ax = axes[0]
    for i, comp in enumerate(['w','x','y','z']):
        ax.plot(t_sim, sim[f'quat_{comp}'], '-', color=colors[i], marker='.', markersize=4, markevery=10, label=f'Sim {comp}')
        ax.plot(t_real, real[f'quat_{comp}'], '--', color=colors[i], marker='o', markersize=3, markevery=15, label=f'Real {comp}')
    ax.set_title('Quaternion Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    common_limits(ax,
                  sim['quat_w']+sim['quat_x']+sim['quat_y']+sim['quat_z'],
                  real['quat_w']+real['quat_x']+real['quat_y']+real['quat_z'])
    ax.legend()

    # 2) Euler (from quat) overlay
    ax = axes[1]
    for i, (comp,label) in enumerate(zip(['x','y','z'], ['roll','pitch','yaw'])):
        ax.plot(t_sim, sim[f'euler_{comp}_quat'], '-', color=colors[i], label=f'Sim {label}')
        ax.plot(t_real, real[f'euler_{comp}_quat'], '--', color=colors[i], label=f'Real {label}')
    ax.set_title('Euler (from quat) Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Degrees', fontsize=10)
    common_limits(ax,
                  sim['euler_x_quat']+sim['euler_y_quat']+sim['euler_z_quat'],
                  real['euler_x_quat']+real['euler_y_quat']+real['euler_z_quat'])
    ax.legend()

    # 3) Gyroscope overlay
    ax = axes[2]
    for i, comp in enumerate(['x','y','z']):
        ax.plot(t_sim, sim[f'gyro_{comp}'], '-', color=colors[i], label=f'Sim gyro_{comp}')
        ax.plot(t_real, real[f'gyro_{comp}'], '--', color=colors[i], label=f'Real gyro_{comp}')
    ax.set_title('Gyroscope Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Rad/s', fontsize=10)
    common_limits(ax,
                  sim['gyro_x']+sim['gyro_y']+sim['gyro_z'],
                  real['gyro_x']+real['gyro_y']+real['gyro_z'])
    ax.legend()

    # 4) Accelerometer overlay
    ax = axes[3]
    for i, comp in enumerate(['x','y','z']):
        ax.plot(t_sim, sim[f'acc_{comp}'], '-', color=colors[i], label=f'Sim acc_{comp}')
        ax.plot(t_real, real[f'acc_{comp}'], '--', color=colors[i], label=f'Real acc_{comp}')
    ax.set_title('Accelerometer Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('m/s²', fontsize=10)
    common_limits(ax,
                  sim['acc_x']+sim['acc_y']+sim['acc_z'],
                  real['acc_x']+real['acc_y']+real['acc_z'])
    ax.legend()

    # 5) Projected Gravity overlay
    ax = axes[4]
    for i, comp in enumerate(['x','y','z']):
        ax.plot(t_sim, sim[f'proj_{comp}'], '-', color=colors[i], label=f'Sim proj_{comp}')
        ax.plot(t_real, real[f'proj_{comp}'], '--', color=colors[i], label=f'Real proj_{comp}')
    ax.set_title('Projected Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Magnitude', fontsize=10)
    common_limits(ax,
                  sim['proj_x']+sim['proj_y']+sim['proj_z'],
                  real['proj_x']+real['proj_y']+real['proj_z'])
    ax.legend()
    

    output_dir = os.path.dirname(real.get('_filename', ''))
    if output_dir:
        save_path = os.path.join(output_dir, f'overlay_{filetitle}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overlay plot to {save_path}")
    
    return fig


def find_avg_freq(sim, real):
    sim_freq = np.mean(sim['loop_freq'])
    real_freq = np.mean(real['loop_freq'])
    return sim_freq, real_freq

def pre_process_data(sim_data, real_data, offset_time):
    #* Truncate the data to the shortest length
    min_len = math.inf
    for key, val in real_data.items():
        if isinstance(val, list):
            min_len = min(min_len, len(val))

    for key, val in sim_data.items():
        if isinstance(val, list):
            min_len = min(min_len, len(val))

    for key, val in real_data.items():
        if isinstance(val, list):
            truncated = len(val) - min_len
            if truncated > 0:
                print(f"Truncating {truncated} values from real data['{key}']")
            real_data[key] = real_data[key][:min_len]

    for key, val in sim_data.items():
        if isinstance(val, list):
            truncated = len(val) - min_len
            if truncated > 0:
                logger.info(f"Truncating {truncated} values from sim data['{key}']")
            sim_data[key] = sim_data[key][:min_len]

    #* Offset the real data timestamps by the specified amount (in seconds)
    if offset_time != 0:
        # Create a copy of the original timestamps
        orig_timestamps = real_data['timestamp'].copy()
        
        # Apply the offset to each timestamp
        for i in range(len(real_data['timestamp'])):
            real_data['timestamp'][i] -= offset_time
            
        logger.info(f"Timestamp range shifted from [{orig_timestamps[0]:.3f}, {orig_timestamps[-1]:.3f}] to [{real_data['timestamp'][0]:.3f}, {real_data['timestamp'][-1]:.3f}]")
    
    return sim_data, real_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--amp', type=float, required=True, help='Amplitude of the sine wave')
    parser.add_argument('--freq', type=float, required=True, help='Frequency of the sine wave')
    args = parser.parse_args()

    # Find data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/amp{args.amp}_freq{args.freq}')
    
    #* Find the latest data files
    # Find real data files
    real_pattern = os.path.join(data_dir, f'real_*.pkl')
    real_files = glob.glob(real_pattern)
    if not real_files:
        raise FileNotFoundError(f"No real data files found matching {real_pattern}")
    
    # Extract timestamps and sort files
    real_file_timestamps = []
    for file in real_files:
        # Extract the timestamp part from the filename
        filename = os.path.basename(file)
        # For filenames like "real_20250424_041312.pkl", extract the full timestamp
        timestamp_str = filename.replace("real_", "").replace(".pkl", "")
        
        # Convert to datetime object for comparison
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        real_file_timestamps.append((file, timestamp))
    
    # Sort by timestamp (newest first)
    real_file_timestamps.sort(key=lambda item: item[1], reverse=True)
    
    # Get the latest file
    real_pickle_file = real_file_timestamps[0][0]
    print(f"Using real data file: {os.path.basename(real_pickle_file)}")
    
    # Find sim data files
    sim_pattern = os.path.join(data_dir, f'sim_*.pkl')
    sim_files = glob.glob(sim_pattern)
    if not sim_files:
        raise FileNotFoundError(f"No simulation data files found matching {sim_pattern}")
    
    # Extract timestamps and sort files
    sim_file_timestamps = []
    for file in sim_files:
        # Extract the timestamp part from the filename
        filename = os.path.basename(file)
        # For filenames like "sim_20250424_040409.pkl", extract the full timestamp
        timestamp_str = filename.replace("sim_", "").replace(".pkl", "")
        
        # Convert to datetime object for comparison
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        sim_file_timestamps.append((file, timestamp))
    
    # Sort by timestamp (newest first)
    sim_file_timestamps.sort(key=lambda item: item[1], reverse=True)
    
    # Get the latest file
    sim_pickle_file = sim_file_timestamps[0][0]
    print(f"Using simulation data file: {os.path.basename(sim_pickle_file)}")

    # Load the pickle files
    with open(real_pickle_file, 'rb') as f:
        real_data = pickle.load(f)
    real_data['_filename'] = real_pickle_file
        
    with open(sim_pickle_file, 'rb') as f:
        sim_data = pickle.load(f)
    sim_data['_filename'] = sim_pickle_file



    sim_data, real_data = pre_process_data(sim_data, real_data, 0.23) #seconds

    sim_freq, real_freq = find_avg_freq(sim_data, real_data)

    plot_title = f'Data at: {sim_freq}hz sim and {real_freq}hz real, Pendulum at: amp{args.amp}deg_freq{args.freq}hz'

    # Generate and save the plots
    plot_side_by_side(sim_data, real_data, plot_title=plot_title, filetitle=f'{args.amp}_{args.freq}')
    plot_overlay(sim_data, real_data, plot_title=plot_title, filetitle=f'{args.amp}_{args.freq}')
    
    print(f"Plots saved to {data_dir}")
