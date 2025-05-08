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

# Local imports
import colorlogging

# Configure logging
logger = logging.getLogger(__name__)
colorlogging.configure()


def plot_accel_xyz(json_file):
    # Load the JSON
    with open(json_file, "r") as f:
        payload = json.load(f)

    os.makedirs(json_file.split(".")[0], exist_ok=True)

    # Extract config information
    config = payload.get("config", {})
    input_type = config.get("input_type", "unknown")
    collected_on = config.get("collected_on", "unknown")
    mode = config.get("mode", "unknown")

    # Create title with config info
    config_title = f"[{mode}] {input_type} ({collected_on})"

    # Extract timestamp and accel data
    data = payload["data"]
    times = [entry["time"] for entry in data]
    # Convert to relative time (seconds since start)
    t0 = times[0]
    times = [t - t0 for t in times]

    ax = [entry["accel_xyz"][0] for entry in data]
    ay = [entry["accel_xyz"][1] for entry in data]
    az = [entry["accel_xyz"][2] for entry in data]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, ax, label="Accel X", color="r")
    plt.plot(times, ay, label="Accel Y", color="g")
    plt.plot(times, az, label="Accel Z", color="b")
    plt.xlabel("Time (s) since start")
    plt.ylabel("Acceleration (m/s²)")
    plt.title(f"Accelerometer X, Y, Z over Time - {config_title}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{json_file.split('.')[0]}/accel_xyz.png")


def plot_projected_gravity(json_file):
    # Load the JSON
    with open(json_file, "r") as f:
        payload = json.load(f)

    # Extract config information
    config = payload.get("config", {})
    input_type = config.get("input_type", "unknown")
    collected_on = config.get("collected_on", "unknown")
    mode = config.get("mode", "unknown")
    actuator_id = config.get("actuator_id", "unknown")
    amplitude = config.get("amplitude", "unknown")
    duration = config.get("duration", "unknown")

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
    # expected_Y = -1 * np.sin(np.radians(act_pos_np))
    expected_Z = -1 * np.sin(np.radians(act_pos_np))

    # Calculate differences
    diff_x = np.abs(proj_x_np - expected_X)
    # diff_y = np.abs(proj_y_np - expected_Y)
    diff_z = np.abs(proj_z_np - expected_Z)

    # Combine X and Z differences
    all_diffs = np.concatenate([diff_x, diff_z])

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
    # ax2.plot(times_np, expected_Y, label='Expected Y', linestyle='--', color=colors[1], linewidth=3)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="path to your data.json")
    args = parser.parse_args()
    json_file = args.json_path
    logger.info(f"Plotting {json_file}")
    plot_accel_xyz(json_file)
    plot_projected_gravity(json_file)
    logger.info(f"Saved {json_file.split('.')[0]}/___.png")
