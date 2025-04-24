import argparse
import csv
import json
import logging
import math
import os
import time
from pathlib import Path
from datetime import datetime
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

import mujoco
import mujoco.viewer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import List


@dataclass
class IMUData:
    amplitude: float
    frequency: float
    timestamp: List[float]
    loop_freq: List[float]
    acc_x: List[float]
    acc_y: List[float]
    acc_z: List[float]
    gyro_x: List[float]
    gyro_y: List[float]
    gyro_z: List[float]
    quat_w: List[float]
    quat_x: List[float]
    quat_y: List[float]
    quat_z: List[float]
    euler_x_quat: List[float]
    euler_y_quat: List[float]
    euler_z_quat: List[float]
    euler_x: List[float]
    euler_y: List[float]
    euler_z: List[float]
    proj_x: List[float]
    proj_y: List[float]
    proj_z: List[float]


def load_model(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model and data from an XML file."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    return model, data


def save_data(data):
    """Save IMU data to a pickle file.
    
    Args:
        data: IMUData object containing all the collected data
    """
    
    # Determine base directory relative to this file and build nested folders: data/YYYYMMDD/imu/{amplitude}_{frequency}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    date_str = datetime.now().strftime("%Y%m%d")
    file_prefix = f"amp{data.amplitude}_freq{data.frequency}"
    dir_path = os.path.join(base_dir, "../data", f"{date_str}_sim")
    os.makedirs(dir_path, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%H%M%S")
    filename = os.path.join(dir_path, f"{file_prefix}_{timestamp}.pkl")
    
    # Convert IMUData object to dictionary
    data_dict = {
        'amplitude': data.amplitude,
        'frequency': data.frequency,
        'timestamp': data.timestamp,
        'loop_freq': data.loop_freq,
        'acc_x': data.acc_x,
        'acc_y': data.acc_y,
        'acc_z': data.acc_z,
        'gyro_x': data.gyro_x,
        'gyro_y': data.gyro_y,
        'gyro_z': data.gyro_z,
        'quat_w': data.quat_w,
        'quat_x': data.quat_x,
        'quat_y': data.quat_y,
        'quat_z': data.quat_z,
        'euler_x_quat': data.euler_x_quat,
        'euler_y_quat': data.euler_y_quat,
        'euler_z_quat': data.euler_z_quat,
        'euler_x': data.euler_x,
        'euler_y': data.euler_y,
        'euler_z': data.euler_z,
        'proj_x': data.proj_x,
        'proj_y': data.proj_y,
        'proj_z': data.proj_z
    }
    
    # Save the data to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    
    logger.info(f"Data saved to {filename}")


def per_loop(data, model, mj_data, sensor_ids):
    """Process data for each simulation step."""
    
    # Get sensor data
    for name, sid in sensor_ids.items():
        if sid < 0:
            continue
        vals = mj_data.sensor(sid).data
        
        if name == "imu_acc":
            data.acc_x.append(float(vals[0]))
            data.acc_y.append(float(vals[1]))
            data.acc_z.append(float(vals[2]))
        elif name == "imu_gyro":
            data.gyro_x.append(float(vals[0]))
            data.gyro_y.append(float(vals[1]))
            data.gyro_z.append(float(vals[2]))
        elif name == "base_link_quat":
            # Extract quaternion
            data.quat_w.append(float(vals[0]))  # w component is first in sensor output
            data.quat_x.append(float(vals[1]))
            data.quat_y.append(float(vals[2]))
            data.quat_z.append(float(vals[3]))
            
            # Convert to Euler angles (in degrees)
            r = Rotation.from_quat([data.quat_x[-1], data.quat_y[-1], data.quat_z[-1], data.quat_w[-1]])
            euler_from_quat = r.as_euler('xyz', degrees=True)
            data.euler_x_quat.append(euler_from_quat[0])
            data.euler_y_quat.append(euler_from_quat[1])
            data.euler_z_quat.append(euler_from_quat[2])
            
            # Calculate projected gravity
            proj_grav_world = r.apply(np.array([0.0, 0.0, 1.0]), inverse=True)
            data.proj_x.append(proj_grav_world[0])
            data.proj_y.append(proj_grav_world[1])
            data.proj_z.append(proj_grav_world[2])
    

    #* There is no direct euler read from Sim
    data.euler_x.append(0.0)
    data.euler_y.append(0.0)
    data.euler_z.append(0.0)



def main():
    parser = argparse.ArgumentParser(description="BIM simulation data collection.")
    parser.add_argument(
        "--model", type=str, default="testbench/scene.xml", help="Path to MuJoCo XML model file"
    )
    parser.add_argument(
        "--body", type=str, default="robot", help="Name of the body to track"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Duration of the experiment (default: 10.0)"
    )
    parser.add_argument(
        "--amplitude", type=float, default=50.0, help="Amplitude of the sine wave in degrees (default: 50.0)"
    )
    parser.add_argument(
        "--frequency", type=float, default=0.5, help="Frequency of the sine wave (default: 0.5)"
    )
    parser.add_argument(
        "--data_freq", type=int, default=100, help="Data frequency in hz (default: 100)"
    )
    parser.add_argument(
        "--disable_simulation", action="store_true", help="Run simulation without visualization"
    )
    parser.add_argument(
        "--joint_name", type=str, default="servo_out", help="Name of the joint/actuator to control"
    )
    args = parser.parse_args()

    # Load the MuJoCo model
    path = Path(args.model)
    model, mj_data = load_model(path)
    model.opt.timestep = 0.001  # 1000 Hz
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

    # Launch passive viewer if simulation is not disabled
    viewer = None
    if not args.disable_simulation:
        viewer = mujoco.viewer.launch_passive(model, mj_data)

    # Precompute IDs for body and sensors
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.body)
    if body_id < 0:
        logger.warning(f"Body '{args.body}' not found")

    # Get sensor IDs
    sensor_list = ["imu_acc", "imu_gyro", "imu_mag", "base_link_quat"]
    sensor_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        for name in sensor_list
    }
    for name, sid in sensor_ids.items():
        if sid >= 0:
            logger.info(f"Sensor '{name}' -> id {sid}")
        else:
            logger.warning(f"Sensor '{name}' not found")

    # Initialize data collection structure
    data = IMUData(
        amplitude=args.amplitude,
        frequency=args.frequency,
        timestamp=[], loop_freq=[],
        acc_x=[], acc_y=[], acc_z=[],
        gyro_x=[], gyro_y=[], gyro_z=[],
        quat_w=[], quat_x=[], quat_y=[], quat_z=[],
        euler_x_quat=[], euler_y_quat=[], euler_z_quat=[],
        euler_x=[], euler_y=[], euler_z=[],
        proj_x=[], proj_y=[], proj_z=[]
    )

    # Main simulation loop
    start_wall = time.time()
    sim_time = 0.0
    step = 0
    per_loop_time = 1.0 / args.data_freq
    steps_per_loop = int(per_loop_time / model.opt.timestep)
    
    logger.info(f"Starting simulation for {args.duration} s...")
    logger.info(f"Collecting data at {args.data_freq} Hz...")
    logger.info(f"Oscillating '{args.joint_name}' at {args.frequency} Hz with amplitude {args.amplitude}°")

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.joint_name)
    if act_id < 0:
        raise ValueError(f"Actuator '{args.joint_name}' not found")

    try:
        while running and sim_time < args.duration:
            cycle_start_time = time.time()

            angle = math.radians(args.amplitude) * math.sin(2 * math.pi * args.frequency * sim_time)
            mj_data.ctrl[act_id] = angle

            # Only collect data at the specified frequency
            if step % steps_per_loop == 0:

                per_loop(data, model, mj_data, sensor_ids)

                data.timestamp.append(time.time() - start_wall)

                # Calculate loop frequency
                current_loop_time = time.time() - cycle_start_time
                if current_loop_time > per_loop_time:
                    logger.warning(f"Loop overrun by {current_loop_time - per_loop_time:.2f} seconds")
                    data.loop_freq.append(1.0 / current_loop_time)
                else:
                    data.loop_freq.append(args.data_freq)

            # Step simulation
            mujoco.mj_step(model, mj_data)
            sim_time += model.opt.timestep
            step += 1

            # Viewer update if enabled
            if viewer and not args.disable_simulation:
                if hasattr(viewer, "sync"):
                    viewer.sync()
                    running = viewer.is_running()
                elif hasattr(viewer, "render"):
                    running = viewer.render()

            # Status logging
            if step % (1000) == 0:  # Log every second (at 1000 Hz)
                logger.info(f"Simulated {sim_time:.2f} s ({step} steps)")

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        save_data(data)
    finally:
        if viewer and hasattr(viewer, "close") and not args.disable_simulation:
            viewer.close()
        
        # Save the collected data
        save_data(data)


if __name__ == "__main__":
    main()
