import asyncio
import time
import pykos
import json
from datetime import datetime
import os
import argparse
import colorlogging
import logging
import numpy as np
import math

logger = logging.getLogger(__name__)
colorlogging.configure()


async def sample_loop(kos, actuator_id, data, test_start, duration, commanded_position_ref, commanded_freq_ref):
    sampling_rate = 100
    end = time.monotonic() + duration
    while time.monotonic() < end:
        t0 = time.monotonic()
        state = await kos.actuator.get_actuators_state([actuator_id])
        data.append((t0 - test_start, state.states[0].position, commanded_position_ref[0], state.states[0].torque, commanded_freq_ref[0]))
        await asyncio.sleep(max(0, 1.0/sampling_rate - (time.monotonic() - t0)))

    
async def run_chirp_test(
    kos, actuator_id, joint_name,
    kp, kd, max_torque, min_pos, max_pos,
    start_freq, end_freq, chirp_duration,
    start_pos, sim, step_size,
    armature, frictionloss, actuatorfrcrange, damping):

    logger.info(f"Running test with kp={kp:.2f}, kd={kd:.2f}")
    
    data = []
    test_start_time = 0
    commanded_position_ref = [start_pos]
    commanded_freq_ref = [start_freq]
    duration = chirp_duration + 1.0 # 0.5 sec for start, 0.5 for end

    logger.info(f"Moving to start position: {start_pos} degrees")
    await kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": start_pos}])
    await asyncio.sleep(1.0)
    
    test_start_time = time.monotonic()
    
    sampler = asyncio.create_task(sample_loop(kos, actuator_id, data, test_start_time, duration, commanded_position_ref, commanded_freq_ref))

    k = (end_freq - start_freq) / duration  # Rate of frequency change
    f0 = start_freq
    
    current_time = time.monotonic() - test_start_time
    while current_time < duration:
        current_time = time.monotonic() - test_start_time
        
        phase = 2.0 * math.pi * (f0 * current_time + 0.5 * k * current_time * current_time)
        
        # Calculate instantaneous frequency and angular velocity
        freq = f0 + k * current_time
        omega = 2.0 * math.pi * freq
        
        # Calculate position and velocity
        amplitude = step_size / 2.0
        position = amplitude * np.sin(phase) + start_pos
        velocity = amplitude * omega * np.cos(phase)

        commanded_position_ref[0] = position
        commanded_freq_ref[0] = freq

        commanded_position = position
        if commanded_position > max_pos:
            commanded_position = max_pos
            logger.warning(f"Commanded position {commanded_position} is greater than max position {max_pos}")
        if commanded_position < min_pos:
            commanded_position = min_pos
            logger.warning(f"Commanded position {commanded_position} is less than min position {min_pos}")
        
        logger.info(f"Time: {current_time:.2f}s, Freq: {freq:.2f}Hz, Position: {position:.2f} degrees")
        
        commands = [
            {
                'actuator_id': actuator_id,
                'position': position,
                'velocity': velocity,
            }
        ]
        
        await kos.actuator.command_actuators(commands)

    # Return to start position
    commanded_position_ref[0] = start_pos
    await kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": 0.0}])
    await asyncio.sleep(1.0)
    
    # Wait for sampling to complete
    await sampler

    # Save collected data to JSON
    simorreal = "sim" if sim else "real"
    fldr_name = f"{datetime.now().strftime('%Y%m%d')}/chirp"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use actual kp and kd values in the filename
    filename = f"{fldr_name}/{simorreal}_{actuator_id}_damp{damping}.json"
    
    # Convert data for JSON serialization
    json_data = [
        {
            "time_since_start": entry[0],
            "position": entry[1],
            "commanded_position": entry[2],
            "torque": entry[3], 
            "freq": entry[4],
        }
        for entry in data
    ]
    
    # Create JSON structure
    output = {
        "config": {
            "kp": kp,
            "kd": kd,
            "max_torque": max_torque,
            "step_size": step_size,
            "chirp_duration": chirp_duration,
            "start_pos": start_pos,
            "start_freq": start_freq,
            "end_freq": end_freq,
            "actuator_id": actuator_id,
            "mode": 'sim' if sim else 'real',
            "armature": armature,
            "frictionloss": frictionloss,
            "actuatorfrcrange": actuatorfrcrange,
            "damping": damping
        },
        "data": json_data
    }
    
    # Write to JSON file
    os.makedirs(fldr_name, exist_ok=True)
    with open(filename, 'w') as jsonfile:
        json.dump(output, jsonfile, indent=2)
    
    logger.info(f"Data saved to {filename}")



async def main():
    kos = pykos.KOS("0.0.0.0")
    
    joint_names = [
        "dof_right_hip_pitch_04",
        "dof_right_hip_roll_03",
        "dof_right_hip_yaw_03",
        "dof_right_knee_04",
        "dof_right_ankle_02",
        # "dof_left_hip_pitch_04",
        # "dof_left_hip_roll_03",
        # "dof_left_hip_yaw_03",
        # "dof_left_knee_04",
        # "dof_left_ankle_02",
        # "dof_right_shoulder_pitch_03",
        # "dof_right_shoulder_roll_03",
        # "dof_right_shoulder_yaw_02",
        # "dof_right_elbow_02",
        # "dof_right_wrist_00",
        # "dof_left_shoulder_pitch_03",
        # "dof_left_shoulder_roll_03",
        # "dof_left_shoulder_yaw_02",
        # "dof_left_elbow_02",
        # "dof_left_wrist_00",
    ]

    for joint_name in joint_names:
        # Setup test configuration
        TEST_CONFIGS = {
            "joint_name": joint_name,
            "min_pos": -30.0,
            "max_pos": 30.0,
            "chirp_duration": 6.0,
            "start_pos": 0.0, #-10 for knee
            "step_size": -10.0,
            "start_freq": 0.2,
            "end_freq": 2.0,
            "sim": args.sim,
        }

        if joint_name == "dof_right_knee_04":
            TEST_CONFIGS["start_pos"] = -10.0

        if joint_name == "dof_left_knee_04":
            TEST_CONFIGS["start_pos"] = 10.0

        if joint_name == "dof_right_elbow_02":
            TEST_CONFIGS["start_pos"] = 10.0

        if joint_name == "dof_left_elbow_02":
            TEST_CONFIGS["start_pos"] = -10.0

        if joint_name == "dof_left_shoulder_roll_03":
            TEST_CONFIGS["start_pos"] = 10.0

        if joint_name == "dof_right_shoulder_roll_03":
            TEST_CONFIGS["start_pos"] = -10.0
        
        
        

        # Read metadata.json to get joint-specific values
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        
        joint_metadata = metadata["joint_name_to_metadata"].get(joint_name)
        
        if not joint_metadata:
            logger.error(f"Joint name {joint_name} not found in metadata.json")
            continue
            
        # Get actuator and passive parameters
        TEST_CONFIGS["kp"] = float(joint_metadata["kp"])
        TEST_CONFIGS["kd"] = float(joint_metadata["kd"])
        TEST_CONFIGS["max_torque"] = float(joint_metadata["max_torque"])
        actuator_id = joint_metadata["id"]
        
        # Get passive parameters
        actuator_type = joint_metadata.get("actuator_type")
        if actuator_type and actuator_type in metadata["actuator_type_passive_param"]:
            passive_params = metadata["actuator_type_passive_param"][actuator_type]
            TEST_CONFIGS["armature"] = float(passive_params["armature"])
            TEST_CONFIGS["frictionloss"] = float(passive_params["frictionloss"])
            TEST_CONFIGS["damping"] = float(passive_params["damping"])
            
            # Parse actuatorfrcrange
            frc_range = passive_params["actuatorfrcrange"].split()
            TEST_CONFIGS["actuatorfrcrange"] = [float(frc_range[0]), float(frc_range[1])]
            
            logger.info(f"Added passive params: armature={TEST_CONFIGS['armature']}, "
                        f"frictionloss={TEST_CONFIGS['frictionloss']}, "
                        f"actuatorfrcrange={TEST_CONFIGS['actuatorfrcrange']}")

        logger.info(f"Base Kp: {TEST_CONFIGS['kp']}, Base Kd: {TEST_CONFIGS['kd']}, Max Torque: {TEST_CONFIGS['max_torque']}")

        for joint_name, config in metadata["joint_name_to_metadata"].items():
            config_actuator_id = config["id"]
            config_kp = float(config["kp"])
            config_kd = float(config["kd"])
            config_max_torque = float(config["max_torque"])
            
            # logger.info(f"Configuring {joint_name} (ID: {actuator_id}) with kp={kp}, kd={kd}, max_torque={max_torque}")
            
            await kos.actuator.configure_actuator(
                actuator_id=config_actuator_id,
                kp=config_kp,
                kd=config_kd,
                max_torque=config_max_torque,
                torque_enabled=True,
            )

      
        # Run the kp/kd sweep
        await run_chirp_test(kos, actuator_id, **TEST_CONFIGS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()
    
    asyncio.run(main())
