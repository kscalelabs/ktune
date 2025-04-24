import pickle
import os
import matplotlib.pyplot as plt
import math

# Use a professional style
plt.style.use('ggplot')

# Path to the pickle file
real_pickle_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/20250423_real/amp50.0_freq0.5_015222.pkl')
sim_pickle_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/20250423_sim/amp50.0_freq0.5_175514.pkl')

"""
# Available keys in the data dictionary:
# 'amplitude', 'frequency', 'timestamp', 'loop_freq', 
# 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
# 'quat_w', 'quat_x', 'quat_y', 'quat_z', 
# 'euler_x_quat', 'euler_y_quat', 'euler_z_quat',
# 'euler_x', 'euler_y', 'euler_z',
# 'proj_x', 'proj_y', 'proj_z'
"""

# Load the pickle file
with open(real_pickle_file, 'rb') as f:
    real_data = pickle.load(f)
    
with open(sim_pickle_file, 'rb') as f:
    sim_data = pickle.load(f)

print(real_data.keys())
print(sim_data.keys())



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
            print(f"Truncating {truncated} values from sim data['{key}']")
        sim_data[key] = sim_data[key][:min_len]

def plot_side_by_side(sim, real):
    """Figure 1: side-by-side subplots"""
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 9), constrained_layout=True)
    t_sim = sim['timestamp']
    t_real = real['timestamp']

    # 1) quaternions (w,x,y,z)
    sim_quats = sim['quat_w'] + sim['quat_x'] + sim['quat_y'] + sim['quat_z']
    real_quats = real['quat_w'] + real['quat_x'] + real['quat_y'] + real['quat_z']
    qmin, qmax = min(sim_quats + real_quats), max(sim_quats + real_quats)

    for ax, data, title in [
        (axes[0,0], sim, 'Simulated Quaternions'),
        (axes[0,1], real, 'Real Quaternions')
    ]:
        ax.plot(t_sim if data is sim else t_real, data['quat_w'], label='w')
        ax.plot(t_sim if data is sim else t_real, data['quat_x'], label='x')
        ax.plot(t_sim if data is sim else t_real, data['quat_y'], label='y')
        ax.plot(t_sim if data is sim else t_real, data['quat_z'], label='z')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Quaternion Value', fontsize=10)
        ax.set_ylim(qmin, qmax)
        ax.grid(True)
        ax.legend()

    # 2) euler angles from quat
    sim_e = sim['euler_x_quat'] + sim['euler_y_quat'] + sim['euler_z_quat']
    real_e = real['euler_x_quat'] + real['euler_y_quat'] + real['euler_z_quat']
    emin, emax = min(sim_e + real_e), max(sim_e + real_e)

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

    # 3) projected gravity
    sim_p = sim['proj_x'] + sim['proj_y'] + sim['proj_z']
    real_p = real['proj_x'] + real['proj_y'] + real['proj_z']
    pmin, pmax = min(sim_p + real_p), max(sim_p + real_p)

    for ax, data, title in [
        (axes[2,0], sim, 'Simulated Projected'),
        (axes[2,1], real, 'Real Projected')
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

def plot_overlay(sim, real):
    """Figure 2: overlay sim vs real"""
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), constrained_layout=True)
    t_sim, t_real = sim['timestamp'], real['timestamp']

    def common_limits(ax, sim_vals, real_vals):
        vmin = min(min(sim_vals), min(real_vals))
        vmax = max(max(sim_vals), max(real_vals))
        ax.set_ylim(vmin, vmax)
        ax.set_xlim(min(t_sim[0], t_real[0]), max(t_sim[-1], t_real[-1]))
        ax.grid(True)

    # 1) quaternions overlay
    ax = axes[0]
    for comp in ['w','x','y','z']:
        ax.plot(t_sim, sim[f'quat_{comp}'], '--', label=f'Sim {comp}')
        ax.plot(t_real, real[f'quat_{comp}'],     '-', label=f'Real {comp}')
    ax.set_title('Quaternion Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    common_limits(ax,
                  sim['quat_w']+sim['quat_x']+sim['quat_y']+sim['quat_z'],
                  real['quat_w']+real['quat_x']+real['quat_y']+real['quat_z'])
    ax.legend()

    # 2) euler_quat overlay
    ax = axes[1]
    for comp,label in zip(['x','y','z'], ['roll','pitch','yaw']):
        ax.plot(t_sim, sim[f'euler_{comp}_quat'], '--', label=f'Sim {label}')
        ax.plot(t_real, real[f'euler_{comp}_quat'],  '-', label=f'Real {label}')
    ax.set_title('Euler (from quat) Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Degrees', fontsize=10)
    common_limits(ax,
                  sim['euler_x_quat']+sim['euler_y_quat']+sim['euler_z_quat'],
                  real['euler_x_quat']+real['euler_y_quat']+real['euler_z_quat'])
    ax.legend()

    # 3) projected overlay
    ax = axes[2]
    for comp in ['x','y','z']:
        ax.plot(t_sim, sim[f'proj_{comp}'], '--', label=f'Sim proj_{comp}')
        ax.plot(t_real, real[f'proj_{comp}'],  '-', label=f'Real proj_{comp}')
    ax.set_title('Projected Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Magnitude', fontsize=10)
    common_limits(ax,
                  sim['proj_x']+sim['proj_y']+sim['proj_z'],
                  real['proj_x']+real['proj_y']+real['proj_z'])
    ax.legend()

if __name__ == '__main__':
    plot_side_by_side(sim_data, real_data)
    plot_overlay(sim_data, real_data)
    plt.show()
