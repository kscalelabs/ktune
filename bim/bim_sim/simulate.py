import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model and data from an XML file."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    return model, data


def simulate(
    model_path: str,
    body_name: str,
    output_file: str,
    duration: float,
    joint_name: str,
    freq: float,
    amplitude_deg: float,
    disable_simulation: bool,
) -> None:
    """
    Run a MuJoCo simulation, collect sensor and kinematic data, and save to CSV.
    
    Args:
        model_path: Path to the MuJoCo XML model file
        body_name: Name of the body to track
        output_file: Path to save the CSV data
        duration: Simulation duration in seconds
        joint_name: Name of the joint/actuator to control
        freq: Oscillation frequency in Hz
        amplitude_deg: Oscillation amplitude in degrees
        disable_simulation: Whether to disable the simulation visualization
    """
    path = Path(model_path)
    model, data = load_model(path)
    model.opt.timestep = 0.001  # 1000 Hz

    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

    # Launch passive viewer if simulation is not disabled
    viewer = None
    if not disable_simulation:
        viewer = mujoco.viewer.launch_passive(model, data)

    # Retrieve actuator ID
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    print(act_id)
    if act_id >= 0:
        logger.info(f"Actuator '{joint_name}' -> id {act_id}")
        amplitude_rad = math.radians(amplitude_deg)
        logger.info(
            f"Oscillating '{joint_name}' at {freq} Hz with ±{amplitude_deg}° "
            f"(±{amplitude_rad:.4f} rad)")
    else:
        logger.warning(f"Actuator '{joint_name}' not found")

    # Precompute IDs for body and sensors
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        logger.warning(f"Body '{body_name}' not found")
    sensor_list = ["imu_acc", "imu_gyro", "imu_mag", "base_link_quat"]
    sensor_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        for name in sensor_list
    }
    for name, sid in sensor_ids.items():
        if sid >= 0:
            logger.info(f"Sensor '{name}' -> id {sid}")

    data_records: list[dict[str, float]] = []
    start_wall = time.time()
    sim_time = 0.0
    step = 0
    logging_interval = 10  # 100 Hz data collection (1000Hz/10)
    display_interval = logging_interval * 10  # Status logging every second
    
    if disable_simulation:
        logger.info("Simulation visualization disabled")
    
    logger.info(f"Starting simulation for {duration} s...")

    try:
        running = True
        while running and sim_time < duration:
            # Actuation logic
            if act_id >= 0:
                angle = amplitude_rad * math.sin(2 * math.pi * freq * sim_time)
                data.ctrl[act_id] = angle

            # Data collection at 100Hz
            if step % logging_interval == 0:
                record = {"timestamp": time.time() - start_wall, "sim_time": sim_time}

                for name, sid in sensor_ids.items():
                    if sid < 0:
                        continue
                    vals = data.sensor(sid).data
                    if name == "imu_acc":
                        record.update({"acc_x": float(vals[0]), "acc_y": float(vals[1]), "acc_z": float(vals[2])})
                    elif name == "imu_gyro":
                        record.update({"gyro_x": float(vals[0]), "gyro_y": float(vals[1]), "gyro_z": float(vals[2])})
                    elif name == "imu_mag":
                        record.update({"mag_x": float(vals[0]), "mag_y": float(vals[1]), "mag_z": float(vals[2])})
                    elif name == "base_link_quat":
                        # Store quaternion components
                        record.update({
                            f"quat_{c}": float(vals[i])
                            for i, c in enumerate(["w", "x", "y", "z"])
                        })
                        
                        # Check if quaternion has non-zero norm before converting
                        quat = [vals[1], vals[2], vals[3], vals[0]]  # [x, y, z, w]
                        quat_norm = np.linalg.norm(quat)
                        
                        if quat_norm > 1e-10:  # Small threshold to check for effectively zero norm
                            # Extract quaternion components - reorder to w, x, y, z for math calculation
                            x, y, z, w = quat  # quat is [x, y, z, w]

                            # Convert quaternion to Euler angles (roll, pitch, yaw)
                            # Roll (x-axis rotation)
                            sinr_cosp = 2 * (w * x + y * z)
                            cosr_cosp = 1 - 2 * (x * x + y * y)
                            roll = math.atan2(sinr_cosp, cosr_cosp)

                            # Pitch (y-axis rotation)
                            sinp = 2 * (w * y - z * x)
                            pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

                            # Yaw (z-axis rotation)
                            siny_cosp = 2 * (w * z + x * y)
                            cosy_cosp = 1 - 2 * (y * y + z * z)
                            yaw = math.atan2(siny_cosp, cosy_cosp)

                            # Convert to degrees
                            roll_deg = math.degrees(roll)
                            pitch_deg = math.degrees(pitch)
                            yaw_deg = math.degrees(yaw)
                            
                            record.update({
                                "base_link_x": float(roll_deg),
                                "base_link_y": float(pitch_deg),
                                "base_link_z": float(yaw_deg)
                            })
                        else:
                            # If quaternion has zero norm, skip Euler conversion
                            logger.warning(f"Skipping Euler conversion at t={sim_time:.3f}: quaternion has zero norm")
                            record.update({
                                "base_link_x": float('nan'),
                                "base_link_y": float('nan'),
                                "base_link_z": float('nan')
                            })

                # Joint position
                if act_id >= 0:
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id >= 0:
                        pos = data.joint(joint_id).qpos[0]
                        record[f"{joint_name}_position"] = float(pos)

                data_records.append(record)

            # Step simulation
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep
            step += 1

            # Viewer update if enabled
            if viewer and not disable_simulation:
                if hasattr(viewer, "sync"):
                    viewer.sync(); running = viewer.is_running()
                elif hasattr(viewer, "render"):
                    running = viewer.render()

            if step % display_interval == 0:
                logger.info(f"Simulated {sim_time:.2f} s ({step} steps)")

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        if viewer and hasattr(viewer, "close") and not disable_simulation:
            viewer.close()

        # Save CSV
        if data_records and output_file:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = sorted(set(k for r in data_records for k in r.keys()))
            # Ensure timestamp first
            if "timestamp" in fieldnames:
                fieldnames.remove("timestamp"); fieldnames.insert(0, "timestamp")
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader(); writer.writerows(data_records)

            wall = time.time() - start_wall
            logger.info(
                f"Saved {len(data_records)} records to '{output_file}' in {wall:.2f} s"
            )

def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo Simulation with data logging"
    )
    parser.add_argument("--model", default="testbench/scene.xml")
    parser.add_argument("--body", default="robot")
    parser.add_argument("--output", default="simulation_data.csv")
    parser.add_argument("--headless", action="store_true", help="Run simulation without visualization")
    args = parser.parse_args()

    simulate(
        model_path=args.model,
        body_name=args.body,
        output_file=args.output,
        duration=5.0,
        joint_name="servo_out",
        freq=0.5,
        amplitude_deg=50.0,
        disable_simulation=args.headless,
    )


if __name__ == "__main__":
    main()
