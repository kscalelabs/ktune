import asyncio
import time
import random
import pykos
import numpy as np
import json
from datetime import datetime
import os
import argparse
import colorlogging
import logging

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
    kos, actuator_id, joint_name,
    kp, kd, max_torque, min_pos, max_pos,
    step_size, step_hold_time, 
    step_count, start_pos, sim,
    armature, frictionloss, actuatorfrcrange, damping):


    kos = pykos.KOS("100.101.101.48") if sim else pykos.KOS("0.0.0.0")
    
    # Initialize data storage
    data = []

    # await kos.actuator.configure_actuator(
    #     actuator_id=actuator_id,
    #     kp=kp,
    #     kd=kd,
    #     max_torque=max_torque,
    #     torque_enabled=True,
    # )

    logger.info(f"Moving to start position: {start_pos} degrees")
    commands = [
        {
            "actuator_id": actuator_id,
            "position": start_pos,
        }
    ]

    await kos.actuator.command_actuators(commands)
    await asyncio.sleep(3.0)

    
    test_start_time = time.monotonic()

    # In the run_step_test function, create a reference that can be updated
    commanded_position_ref = [start_pos]  # Using a list as a mutable reference

    duration = step_hold_time + 3.0
    target_pos = start_pos + step_size

    logger.info(f"Going to target position: {target_pos}")

    if target_pos > max_pos:
        target_pos = start_pos
        logger.warning(f"Rejected, Target position {target_pos} is greater than max position {max_pos}")
    elif target_pos < min_pos:
        target_pos = start_pos
        logger.warning(f"Rejected, Target position {target_pos} is less than min position {min_pos}")

    sampler = asyncio.create_task(sample_loop(kos, actuator_id, data, test_start_time, duration, commanded_position_ref))
    
    await asyncio.sleep(1.0)

    commanded_position_ref[0] = target_pos

    commands = [
        {
            'actuator_id': actuator_id,
            'position': target_pos,
        }
    ]
    await kos.actuator.command_actuators(commands)
    await asyncio.sleep(step_hold_time)

    commanded_position_ref[0] = start_pos

    commands = [
        {
            'actuator_id': actuator_id,
            'position': 0.0,
        }
    ]
    await kos.actuator.command_actuators(commands)

    await asyncio.sleep(1.0)
    await sampler


    # Save collected data to JSON
    simorreal = "sim" if sim else "real"
    fldr_name = f"{datetime.now().strftime('%Y%m%d')}/step"
    filename = f"{fldr_name}/{simorreal}_{actuator_id}_damp{damping}.json"
    
    # Convert data to list of dictionaries for JSON serialization
    json_data = []
    for entry in data:
        json_data.append({
            "time_since_start": entry[0],
            "position": entry[1],
            "commanded_position": entry[2],
            "torque": entry[3]
        })
    
    # Create JSON structure with config as header and data points
    output = {
        "config": {
            "kp": kp,
            "kd": kd,
            "max_torque": max_torque,
            "step_size": step_size,
            "step_hold_time": step_hold_time,
            "step_count": step_count,
            "start_pos": start_pos,
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
        TEST_CONFIGS = {
            "joint_name": joint_name,
            "min_pos": -30.0,
            "max_pos": 30.0,

            "step_hold_time": 2.0, # seconds
            "step_count": 100,  # 1000
            "start_pos": 0.0,       # degrees

            "step_size": -10.0,       # degrees

            "sim": args.sim,
        }

        if joint_name == "dof_right_knee_04":
            TEST_CONFIGS["step_size"] = -10.0

        if joint_name == "dof_left_knee_04":
            TEST_CONFIGS["step_size"] = 10.0

        if joint_name == "dof_right_elbow_02":
            TEST_CONFIGS["step_size"] = 10.0

        if joint_name == "dof_left_elbow_02":
            TEST_CONFIGS["step_size"] = -10.0

        if joint_name == "dof_left_shoulder_roll_03":
            TEST_CONFIGS["step_size"] = 10.0

        if joint_name == "dof_right_shoulder_roll_03":
            TEST_CONFIGS["step_size"] = -10.0

        if joint_name == "dof_left_hip_roll_03":
            TEST_CONFIGS["step_size"] = 10.0
        
        if joint_name == "dof_left_shoulder_roll_03":
            TEST_CONFIGS["step_size"] = 10.0
    
        
        

        # Read metadata.json to get joint-specific kp, kd, and max_torque values
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        
        joint_name = TEST_CONFIGS["joint_name"]
        joint_metadata = metadata["joint_name_to_metadata"].get(joint_name)
        
        if joint_metadata:
            TEST_CONFIGS["kp"] = float(joint_metadata["kp"])
            TEST_CONFIGS["kd"] = float(joint_metadata["kd"])
            TEST_CONFIGS["max_torque"] = float(joint_metadata["max_torque"])
            actuator_id = joint_metadata["id"]
            
            # Get actuator type and passive parameters
            actuator_type = joint_metadata.get("actuator_type")
            if actuator_type and actuator_type in metadata["actuator_type_passive_param"]:
                passive_params = metadata["actuator_type_passive_param"][actuator_type]
                TEST_CONFIGS["armature"] = float(passive_params["armature"])
                TEST_CONFIGS["frictionloss"] = float(passive_params["frictionloss"])
                TEST_CONFIGS["damping"] = float(passive_params["damping"])
                
                # Parse actuatorfrcrange (a space-separated string)
                frc_range = passive_params["actuatorfrcrange"].split()
                TEST_CONFIGS["actuatorfrcrange"] = [float(frc_range[0]), float(frc_range[1])]
                
                logger.info(f"Added passive params: armature={TEST_CONFIGS['armature']}, "
                            f"frictionloss={TEST_CONFIGS['frictionloss']}, "
                            f"actuatorfrcrange={TEST_CONFIGS['actuatorfrcrange']}")
        else:
            logger.error(f"Joint name {joint_name} not found in metadata.json")
            exit(1)

        logger.info(f"Kp: {TEST_CONFIGS['kp']}, Kd: {TEST_CONFIGS['kd']}, Max Torque: {TEST_CONFIGS['max_torque']}")
        
        for config_joint_name, config_joint_metadata in metadata["joint_name_to_metadata"].items():
            config_actuator_id = config_joint_metadata["id"]
            config_kp = float(config_joint_metadata["kp"])
            config_kd = float(config_joint_metadata["kd"])
            config_max_torque = float(config_joint_metadata["max_torque"])
            
            # logger.info(f"Configuring {joint_name} (ID: {actuator_id}) with kp={kp}, kd={kd}, max_torque={max_torque}")
            
            await kos.actuator.configure_actuator(
                actuator_id=config_actuator_id,
                kp=config_kp,
                kd=config_kd,
                max_torque=config_max_torque,
                torque_enabled=True,
            )


        asyncio.run(run_step_test(kos, actuator_id, **TEST_CONFIGS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    asyncio.run(main())


