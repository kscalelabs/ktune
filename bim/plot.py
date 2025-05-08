# Standard library imports
import argparse
import json
import logging
import os
import glob

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import xax
from scipy.signal import medfilt

# Local imports
import colorlogging

# Configure logging
logger = logging.getLogger(__name__)
colorlogging.configure()


def plot_accel_xyz(json_file, imu_loc):
    # Load the JSON
    with open(json_file, "r") as f:
        payload = json.load(f)

    os.makedirs(json_file.split(".")[0], exist_ok=True)

    # Extract config information
    config = payload.get("config", {})
    input_type = config.get("input_type", "")
    collected_on = config.get("collected_on", "")
    mode = config.get("mode", "")
    data_entries = payload["data"]

    times = [entry["time"] for entry in data_entries]
    t0 = times[0]
    times = [t - t0 for t in times]

    accel_x = []
    accel_y = []
    accel_z = []
    act_pos = []
    for entry in data_entries:
        accel_x.append(entry["accel_xyz"][0])
        accel_y.append(entry["accel_xyz"][1])
        accel_z.append(entry["accel_xyz"][2])
        act_pos.append(entry["position"])

    # Convert lists to numpy arrays for calculations
    times_np = np.array(times)
    accel_x_np = np.array(accel_x)
    accel_y_np = np.array(accel_y)
    accel_z_np = np.array(accel_z)
    act_pos_np = np.array(act_pos)

    # Create title with config info
    config_title = f"[{mode}] {input_type} ({collected_on})"

    expected_X = -10 * np.cos(np.radians(act_pos_np))

    if imu_loc == "front":
        expected_Z = np.zeros_like(act_pos_np)
        if mode == "sim":
            #! Temp fix, MJCF orientation is wrong
            expected_Y = 10 * np.sin(np.radians(act_pos_np))
        else:
            expected_Y = -10 * np.sin(np.radians(act_pos_np))
    elif imu_loc == "right":
        expected_Y = np.zeros_like(act_pos_np)
        expected_Z = 10 * np.sin(np.radians(act_pos_np))
    else:
        raise ValueError(f"Invalid IMU location: {imu_loc}")


    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # First subplot with all data and trend lines
    ax1.plot(times_np, accel_x_np, label="Accel X", color="r", linestyle="-.")
    ax1.plot(times_np, accel_y_np, label="Accel Y", color="g", linestyle="-.")
    ax1.plot(times_np, accel_z_np, label="Accel Z", color="b", linestyle="-.")
    ax1.plot(times_np, expected_X, label="Expected X", color="r", linestyle="dashed", linewidth=1)
    ax1.plot(times_np, expected_Y, label="Expected Y", color="g", linestyle="dashed", linewidth=1)
    ax1.plot(times_np, expected_Z, label="Expected Z", color="b", linestyle="dashed", linewidth=1)
    
    # Calculate trends using scipy.signal.medfilt (median filter)
    # Use median filter to create step-like trends (window size can be adjusted)
    window_size = max(3, len(times_np) // 15)  # Make window size odd
    if window_size % 2 == 0:
        window_size += 1
    
    x_trend = medfilt(accel_x_np, window_size)
    y_trend = medfilt(accel_y_np, window_size) 
    z_trend = medfilt(accel_z_np, window_size)
    
    # Plot the trend lines
    if mode != "sim":  # Skip X trend plot for sim mode
        ax1.plot(times_np, x_trend, label="X Trend", color="k", linestyle="-", linewidth=2)
        ax1.plot(times_np, y_trend, label="Y Trend", color="k", linestyle="-", linewidth=2)
        ax1.plot(times_np, z_trend, label="Z Trend", color="k", linestyle="-", linewidth=2)
    
    # Calculate MSE (from expected theoretical values)
    mse_x = np.mean((accel_x_np - expected_X) ** 2)
    mse_y = np.mean((accel_y_np - expected_Y) ** 2)
    mse_z = np.mean((accel_z_np - expected_Z) ** 2)
    mse_all = np.mean([mse_x, mse_y, mse_z])
    
    # Calculate standard deviation from trend lines (instead of variance)
    std_x = np.std(accel_x_np - x_trend)
    std_y = np.std(accel_y_np - y_trend)
    std_z = np.std(accel_z_np - z_trend)
    std_all = np.mean([std_x, std_y, std_z])
    
    # Add text box with metrics
    textstr = (f"MSE from expected values:\n"
               f"X: {mse_x:.2f}, Y: {mse_y:.2f}, Z: {mse_z:.2f}\n"
               f"Avg: {mse_all:.2f}\n\n"
               f"Standard deviation from \n trend (median filter):\n"
               f"X: {std_x:.2f}, Y: {std_y:.2f}, Z: {std_z:.2f}\n"
               f"Avg: {std_all:.2f}")
    
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax1.text(
        0.02,
        0.15,
        textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=props,
    )
    
    ax1.set_xlabel("Time (s) since start")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.set_title(f"Accelerometer X, Y, Z with Expected Values - {config_title}")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Second subplot with just accelerometer data
    ax2.plot(times_np, accel_x_np, label="Accel X", color="r", linestyle="dotted")
    ax2.plot(times_np, accel_y_np, label="Accel Y", color="g", linestyle="dotted")
    ax2.plot(times_np, accel_z_np, label="Accel Z", color="b", linestyle="dotted")
    
    # Add text box showing number of data points
    data_points_text = f"Number of data points: {len(accel_y_np)}"
    data_props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax2.text(
        0.01,
        0.4,
        data_points_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=data_props,
    )
    
    ax2.set_xlabel("Time (s) since start")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.set_title(f"Accelerometer X, Y, Z Data - {config_title}")
    ax2.legend(loc="upper left")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{json_file.split('.')[0]}/accel_xyz.png")


def plot_projected_gravity(json_file, imu_loc):
    # Load the JSON
    with open(json_file, "r") as f:
        payload = json.load(f)

    # Extract config information
    config = payload.get("config", {})
    input_type = config.get("input_type", "")
    collected_on = config.get("collected_on", "")
    mode = config.get("mode", "")
    actuator_id = config.get("actuator_id", "")
    amplitude = config.get("amplitude", "")
    duration = config.get("duration", "")

    # Create title with config info
    config_title = (
        f"{mode} - {input_type} of {amplitude} for {duration} seconds ({collected_on})"
    )

    data_entries = payload["data"]
    times = [entry["time"] for entry in data_entries]
    t0 = times[0]
    times = [t - t0 for t in times]

    proj_x = []
    proj_y = []
    proj_z = []
    act_pos = []
    logger.info(f"XAX taking time.")
    for entry in data_entries:
        quat_xyzw = entry["quat_xyzw"]
        # Convert from [x,y,z,w] to [w,x,y,z] for the function
        q = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        projected_gravity = xax.get_projected_gravity_vector_from_quat(q)
        proj_x.append(projected_gravity[0])
        proj_y.append(projected_gravity[1])
        proj_z.append(projected_gravity[2])
        act_pos.append(entry["position"])

    # Convert lists to numpy arrays for calculations
    times_np = np.array(times)
    proj_x_np = np.array(proj_x)
    proj_y_np = np.array(proj_y)
    proj_z_np = np.array(proj_z)
    act_pos_np = np.array(act_pos)

    # Calculate expected gravity based on actuator position
    expected_X = -1 * np.cos(np.radians(act_pos_np))

    if imu_loc == "front":
        expected_Z = np.zeros_like(act_pos_np)
        if mode == "sim":
            #! Temp fix, MJCF orientation is wrong
            expected_Y = 1 * np.sin(np.radians(act_pos_np))
        else:
            expected_Y = -1 * np.sin(np.radians(act_pos_np))
    elif imu_loc == "right":
        expected_Y = np.zeros_like(act_pos_np)
        expected_Z = np.sin(np.radians(act_pos_np))
    else:
        raise ValueError(f"Invalid IMU location: {imu_loc}")

    # Calculate differences
    diff_x = np.abs(proj_x_np - expected_X)
    diff_y = np.abs(proj_y_np - expected_Y)
    diff_z = np.abs(proj_z_np - expected_Z)

    # Combine X and Z differences
    all_diffs = np.concatenate([diff_x, diff_y, diff_z])

    # Calculate metrics
    avg_diff = np.mean(all_diffs)
    max_diff = np.max(all_diffs)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 2]}
    )

    # Plot actuator position in top subplot
    ax1.plot(
        times_np, act_pos_np, label="Actuator Position", linewidth=2, color="purple"
    )
    ax1.set_xlabel("Time (s) since start")
    ax1.set_ylabel("Actuator Position")
    ax1.set_title(f"Actuator Position Over Time - {config_title}")
    ax1.grid(True)

    # Plot projected gravity comparison in bottom subplot
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red

    ax2.plot(
        times_np,
        expected_X,
        label="Expected X",
        linestyle="--",
        color=colors[0],
        linewidth=3,
    )
    ax2.plot(
        times_np,
        expected_Y,
        label="Expected Y",
        linestyle="--",
        color=colors[1],
        linewidth=3,
    )
    ax2.plot(
        times_np,
        expected_Z,
        label="Expected Z",
        linestyle="--",
        color=colors[2],
        linewidth=3,
    )
    ax2.plot(times_np, proj_x_np, label="Projected X", color=colors[0], linewidth=2)
    ax2.plot(times_np, proj_y_np, label="Projected Y", color=colors[1], linewidth=2)
    ax2.plot(times_np, proj_z_np, label="Projected Z", color=colors[2], linewidth=2)

    # Add text box with metrics
    textstr = f"Difference across X and Z,\n  multiplied by 100 \n Avg Diff (e-2): {avg_diff*100:.2f}\nMax Diff (e-2): {max_diff*100:.2f}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax2.text(
        0.80,
        0.75,
        textstr,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    ax2.set_xlabel("Time (s) since start")
    ax2.set_ylabel("Projected Gravity")
    ax2.set_title(f"Position Calculated vs Measured Projected Gravity - {config_title}")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{json_file.split('.')[0]}/proj_gravity.png")


def add_noise_np(
    observation: np.ndarray,
    rng: np.random.Generator,
    noise_type: str,
    noise_level: float,
    curriculum_level: float,
) -> np.ndarray:
    """Add Gaussian or uniform noise to a NumPy array."""
    if noise_type == "gaussian":
        # standard_normal produces samples ~ N(0,1)
        noise = rng.standard_normal(observation.shape)
    elif noise_type == "uniform":
        # random() in [0,1), so 2*rng.random-1 gives uniform in [-1,1)
        noise = (rng.random(observation.shape) * 2) - 1
    else:
        raise ValueError(f"Invalid noise type: {noise_type!r}")
    return observation + noise * noise_level * curriculum_level


def plot_accel_xyz_with_numpy_noise(
    json_file: str,
    noise_level: float = 0.5,
    noise_type: str = "gaussian",
    curriculum_level: float = 1.0,
    seed: int = 0,
    imu_loc: str = "front",
):
    # --- LOAD YOUR COLLECTED DATA ---
    with open(json_file, "r") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    input_type = config.get("input_type", "unknown")
    collected_on = config.get("collected_on", "unknown")
    mode = config.get("mode", "unknown")

    if mode == "real":
        return

    data_entries = payload["data"]
    times = [entry["time"] for entry in data_entries]
    t0 = times[0]
    times = [t - t0 for t in times]

    # Extract position data
    act_pos = [entry["position"] for entry in data_entries]

    # stack accel into an (N,3) array
    accel = np.vstack([
        [d["accel_xyz"][i] for d in data_entries]
        for i in range(3)
    ]).T  # shape = (N, 3)

    # --- ADD NOISE WITH NUMPY ---
    rng = np.random.default_rng(seed)
    noisy_accel = add_noise_np(accel, rng, noise_type, noise_level, curriculum_level)

    # Convert to numpy arrays
    times_np = np.array(times)
    act_pos_np = np.array(act_pos)
    ax_vals, ay_vals, az_vals = noisy_accel.T

    # Calculate expected values
    expected_X = -10 * np.cos(np.radians(act_pos_np))
    if imu_loc == "front":
        expected_Z = np.zeros_like(act_pos_np)
        expected_Y = 10 * np.sin(np.radians(act_pos_np))
    elif imu_loc == "right":
        expected_Y = np.zeros_like(act_pos_np)
        expected_Z = 10 * np.sin(np.radians(act_pos_np))
    else:
        raise ValueError(f"Invalid IMU location: {imu_loc}")

    # Create title with config info
    config_title = f"[{mode}] {input_type} ({collected_on})"
    
    out_dir = json_file.rsplit(".", 1)[0]
    os.makedirs(out_dir, exist_ok=True)

    # Create figure with subplots (matching plot_accel_xyz structure)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # First subplot with all data and expected values
    ax1.plot(times_np, ax_vals, label="Accel X (noisy)", color="r", linestyle="-.")
    ax1.plot(times_np, ay_vals, label="Accel Y (noisy)", color="g", linestyle="-.")
    ax1.plot(times_np, az_vals, label="Accel Z (noisy)", color="b", linestyle="-.")
    ax1.plot(times_np, expected_X, label="Expected X", color="r", linestyle="dashed", linewidth=1)
    ax1.plot(times_np, expected_Y, label="Expected Y", color="g", linestyle="dashed", linewidth=1)
    ax1.plot(times_np, expected_Z, label="Expected Z", color="b", linestyle="dashed", linewidth=1)
    
    # Calculate trends using median filter
    window_size = max(3, len(times_np) // 15)
    if window_size % 2 == 0:
        window_size += 1
    
    y_trend = medfilt(ay_vals, window_size) 
    z_trend = medfilt(az_vals, window_size)
    x_trend = medfilt(ax_vals, window_size)
    
    ax1.plot(times_np, y_trend, label="Y Trend", color="k", linestyle="-", linewidth=2)
    ax1.plot(times_np, z_trend, label="Z Trend", color="k", linestyle="-", linewidth=2)
    ax1.plot(times_np, x_trend, label="X Trend", color="k", linestyle="-", linewidth=2)
    # Calculate MSE
    mse_x = np.mean((ax_vals - expected_X) ** 2)
    mse_y = np.mean((ay_vals - expected_Y) ** 2)
    mse_z = np.mean((az_vals - expected_Z) ** 2)
    mse_all = np.mean([mse_x, mse_y, mse_z])
    
    # Calculate standard deviation
    x_trend = medfilt(ax_vals, window_size)  # Calculate x_trend for std calculation
    std_x = np.std(ax_vals - x_trend)
    std_y = np.std(ay_vals - y_trend)
    std_z = np.std(az_vals - z_trend)
    std_all = np.mean([std_x, std_y, std_z])
    
    # Add text box with metrics
    textstr = (f"MSE from expected values:\n"
               f"X: {mse_x:.2f}, Y: {mse_y:.2f}, Z: {mse_z:.2f}\n"
               f"Avg: {mse_all:.2f}\n\n"
               f"Standard deviation from \n trend (median filter):\n"
               f"X: {std_x:.2f}, Y: {std_y:.2f}, Z: {std_z:.2f}\n"
               f"Avg: {std_all:.2f}")
    
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax1.text(
        0.02,
        0.15,
        textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=props,
    )
    
    ax1.set_xlabel("Time (s) since start")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.set_title(f"Accelerometer with K-Sim Noise Added, compare to expected - {config_title}")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Second subplot with just accelerometer data
    ax2.plot(times_np, ax_vals, label="Accel X (noisy)", color="r", linestyle="dotted")
    ax2.plot(times_np, ay_vals, label="Accel Y (noisy)", color="g", linestyle="dotted")
    ax2.plot(times_np, az_vals, label="Accel Z (noisy)", color="b", linestyle="dotted")
    
    # Add text box showing number of data points
    data_points_text = f"Number of data points: {len(ay_vals)}"
    data_props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax2.text(
        0.01,
        0.4,
        data_points_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=data_props,
    )
    
    ax2.set_xlabel("Time (s) since start")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.set_title(f"Accelerometer Data with K-Sim Noise Added - {config_title}")
    ax2.legend(loc="upper left")
    ax2.grid(True)
    
    plt.tight_layout()

    out_path = os.path.join(out_dir, "accel_xyz_noisy.png")
    plt.savefig(out_path)
    print(f"Saved noisy plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="path to your data.json")
    parser.add_argument("--imu_loc", choices=["front", "right"], default="front", help="location of the IMU")
    args = parser.parse_args()
    json_file = args.json_path
    logger.info(f"Plotting {json_file}")
    plot_accel_xyz(json_file, args.imu_loc)
    plot_projected_gravity(json_file, args.imu_loc)
    plot_accel_xyz_with_numpy_noise(json_file, imu_loc=args.imu_loc)
    logger.info(f"Saved {json_file.split('.')[0]}/___.png")
