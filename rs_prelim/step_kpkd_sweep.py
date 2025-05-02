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

logger = logging.getLogger(__name__)
colorlogging.configure()


async def sample_loop(kos, actuator_id, data, test_start, duration, commanded_position_ref):
    sampling_rate = 100
    end = time.monotonic() + duration
    while time.monotonic() < end:
        t0 = time.monotonic()
        state = await kos.actuator.get_actuators_state([actuator_id])
        data.append((t0 - test_start, state.states[0].position, commanded_position_ref[0], state.states[0].torque))
        await asyncio.sleep(max(0, 1.0/sampling_rate - (time.monotonic() - t0)))


async def run_step_test(
    actuator_id, joint_name,
    kp, kd, max_torque, min_pos, max_pos,
    step_size, step_hold_time, 
    start_pos, sim,
    armature, frictionloss, actuatorfrcrange):

    logger.info(f"Running test with kp={kp:.2f}, kd={kd:.2f}")

    kos = pykos.KOS("0.0.0.0")
    
    # Initialize data storage
    data = []
    test_start_time = 0
    commanded_position_ref = [start_pos]
    duration = step_hold_time + 3.0
    target_pos = start_pos + step_size

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
    await asyncio.sleep(3.0)
    
    # Run the test
    test_start_time = time.monotonic()
    
    # Validate target position
    if target_pos > max_pos:
        target_pos = start_pos
        logger.warning(f"Rejected, Target position {target_pos} is greater than max position {max_pos}")
    elif target_pos < min_pos:
        target_pos = start_pos
        logger.warning(f"Rejected, Target position {target_pos} is less than min position {min_pos}")

    # Start sampling
    sampler = asyncio.create_task(sample_loop(kos, actuator_id, data, test_start_time, duration, commanded_position_ref))
    await asyncio.sleep(1.0)

    # Move to target position
    commanded_position_ref[0] = target_pos
    await kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": target_pos}])
    await asyncio.sleep(step_hold_time)

    # Return to start position
    commanded_position_ref[0] = start_pos
    await kos.actuator.command_actuators([{"actuator_id": actuator_id, "position": 0.0}])
    await asyncio.sleep(1.0)
    
    # Wait for sampling to complete
    await sampler

    # Save collected data to JSON
    simorreal = "sim" if sim else "real"
    fldr_name = datetime.now().strftime("%Y%m%d")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use actual kp and kd values in the filename
    filename = f"{fldr_name}/{simorreal}_{actuator_id}_kp{kp:.2f}_kd{kd:.2f}_{timestamp_str}.json"
    
    # Convert data for JSON serialization
    json_data = [
        {
            "time_since_start": entry[0],
            "position": entry[1],
            "commanded_position": entry[2],
            "torque": entry[3]
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
            "step_hold_time": step_hold_time,
            "start_pos": start_pos,
            "actuator_id": actuator_id,
            "mode": 'sim' if sim else 'real',
            "armature": armature,
            "frictionloss": frictionloss,
            "actuatorfrcrange": actuatorfrcrange
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
    
    kp_values = [base_kp, 130, 110, 70, 50, 30, 160, 170]
    kd_values = [base_kd, 7, 6, 5, 4, 9, 10]
    
    logger.info(f"Starting kp/kd sweep with base values: kp={base_kp:.2f}, kd={base_kd:.2f}")
    logger.info(f"Testing kp range: {min(kp_values):.2f} to {max(kp_values):.2f}")
    logger.info(f"Testing kd range: {min(kd_values):.2f} to {max(kd_values):.2f}")
    
    # Run through all combinations of kp and kd values
    for kp in kp_values:
        for kd in kd_values:
            logger.info(f"Testing kp={kp:.2f}, kd={kd:.2f}")
            
            # Run the step test with the current values
            await run_step_test(
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
    TEST_CONFIGS = {
        "joint_name": "dof_right_hip_pitch_04",
        "min_pos": -30.0,
        "max_pos": 30.0,
        "step_hold_time": 2.0,
        "start_pos": 0.0,
        "step_size": -10.0,
        "sim": args.sim,
    }

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
        
        # Parse actuatorfrcrange
        frc_range = passive_params["actuatorfrcrange"].split()
        TEST_CONFIGS["actuatorfrcrange"] = [float(frc_range[0]), float(frc_range[1])]
        
        logger.info(f"Added passive params: armature={TEST_CONFIGS['armature']}, "
                    f"frictionloss={TEST_CONFIGS['frictionloss']}, "
                    f"actuatorfrcrange={TEST_CONFIGS['actuatorfrcrange']}")

    logger.info(f"Base Kp: {TEST_CONFIGS['kp']}, Base Kd: {TEST_CONFIGS['kd']}, Max Torque: {TEST_CONFIGS['max_torque']}")

    # Run the kp/kd sweep
    asyncio.run(run_kpkd_sweep(actuator_id, TEST_CONFIGS))
