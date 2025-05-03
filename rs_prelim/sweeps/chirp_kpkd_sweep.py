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
    actuator_id, joint_name,
    kp, kd, max_torque, min_pos, max_pos,
    start_freq, end_freq, chirp_duration,
    start_pos, sim, step_size,
    armature, frictionloss, actuatorfrcrange, damping):

    logger.info(f"Running test with kp={kp:.2f}, kd={kd:.2f}")

    kos = pykos.KOS("0.0.0.0")
    
    # Initialize data storage
    data = []
    test_start_time = 0
    commanded_position_ref = [start_pos]
    commanded_freq_ref = [start_freq]
    duration = chirp_duration + 1.0 # 0.5 sec for start, 0.5 for end

    # Configure the actuator
    await kos.actuator.configure_actuator(
        actuator_id=actuator_id,
        kp=kp,
        kd=kd,
        max_torque=max_torque,
        torque_enabled=True,
    )

    # Move to start position
    logger.info(f"Moving to start position: {start_pos} degrees")
    await kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": start_pos}])
    await asyncio.sleep(1.0)
    
    # Run the test
    test_start_time = time.monotonic()
    
    # Start sampling
    sampler = asyncio.create_task(sample_loop(kos, actuator_id, data, test_start_time, duration, commanded_position_ref, commanded_freq_ref))

     # Calculate chirp sweep rate based on start and end frequencies and duration
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
    fldr_name = f"data/sim_data/f{frictionloss}_d{damping}/chirp"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use actual kp and kd values in the filename
    filename = f"{fldr_name}/{actuator_id}_kp{kp:.2f}_kd{kd:.2f}.json"
    
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


async def run_kpkd_sweep(actuator_id, config):
    """Run the step test with different kp and kd values."""
    
    base_kp = config.pop("kp")
    base_kd = config.pop("kd")
    
    # kp_values = [base_kp, 130, 110, 70, 50, 30, 160, 170]
    # kd_values = [base_kd, 7, 6, 5, 4, 9, 10]
    kp_values = [base_kp]
    kd_values = [base_kd]
    
    logger.info(f"Starting kp/kd sweep with base values: kp={base_kp:.2f}, kd={base_kd:.2f}")
    logger.info(f"Testing kp range: {min(kp_values):.2f} to {max(kp_values):.2f}")
    logger.info(f"Testing kd range: {min(kd_values):.2f} to {max(kd_values):.2f}")
    
    # Run through all combinations of kp and kd values
    for kp in kp_values:
        for kd in kd_values:
            logger.info(f"Testing kp={kp:.2f}, kd={kd:.2f}")
            
            # Run the step test with the current values
            await run_chirp_test(
                actuator_id=actuator_id, 
                kp=kp,
                kd=kd,
                **config
            )
            
            # Brief pause between tests
            await asyncio.sleep(1.0)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()
    
    # Setup test configuration
    for joint_name in ["dof_right_hip_pitch_04", "dof_right_hip_roll_03", "dof_right_hip_yaw_03", "dof_right_knee_04", "dof_right_ankle_02"]:
        TEST_CONFIGS = {
            "joint_name": joint_name,
            "min_pos": -30.0,
            "max_pos": 30.0,
            "chirp_duration": 6.0,
            "start_pos": 0.0,
            "step_size": -10.0,
            "start_freq": 0.2,
            "end_freq": 1.5,
            "sim": args.sim,
        }

        if joint_name == "dof_right_knee_04":
            TEST_CONFIGS["start_pos"] = -10.0

        # Read metadata.json to get joint-specific values
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        
        joint_name = TEST_CONFIGS["joint_name"]
        joint_metadata = metadata["joint_name_to_metadata"].get(joint_name)
        
        if not joint_metadata:
            logger.error(f"Joint name {joint_name} not found in metadata.json")
            exit(1)
            
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

        # Run the kp/kd sweep
        asyncio.run(run_kpkd_sweep(actuator_id, TEST_CONFIGS))
