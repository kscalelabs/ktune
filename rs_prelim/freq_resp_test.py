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
import math

logger = logging.getLogger(__name__)

colorlogging.configure()


async def sample_loop(kos, actuator_id, DATA, test_start, duration, commanded_position_ref):
    sampling_rate = 100
    end = time.monotonic() + duration
    while time.monotonic() < end:
        t0 = time.monotonic()
        state = await kos.actuator.get_actuators_state([actuator_id])
        DATA.append((t0 - test_start, state.states[0].position, commanded_position_ref[0], state.states[0].torque))
        await asyncio.sleep(max(0, 1.0/sampling_rate - (time.monotonic() - t0)))


async def run_test(
    wave_type, kos, actuator_id, 
    simorreal, collected_at,
    joint_name,
    kp, kd, max_torque,
    step_size, step_hold_time, 
    start_freq, end_freq, chirp_duration,
    step_count, start_pos,
    armature, frictionloss, actuatorfrcrange, damping):

    kos = pykos.KOS("0.0.0.0")
    
    DATA = []

    await kos.actuator.configure_actuator(
        actuator_id=actuator_id,
        kp=kp,
        kd=kd,
        max_torque=max_torque,
        torque_enabled=True,
    )

    logger.info(f"Configured actuator {actuator_id} with kp={kp}, kd={kd}, max_torque={max_torque}")

    commands = [
        {
            "actuator_id": actuator_id,
            "position": start_pos,
        }
    ]

    await kos.actuator.command_actuators(commands)
    await asyncio.sleep(2.0)

    
    test_start_time = time.monotonic()

    commanded_position_ref = [start_pos] 

    duration = step_hold_time + 3.0
    target_pos = start_pos + step_size


    sampler = asyncio.create_task(sample_loop(kos, actuator_id, DATA, test_start_time, duration, commanded_position_ref))
    await asyncio.sleep(1.0)

    if wave_type == "step":
        commanded_position_ref[0] = target_pos

        commands = [
            {
                'actuator_id': actuator_id,
                'position': target_pos,
            }
        ]
        await kos.actuator.command_actuators(commands)
        await asyncio.sleep(step_hold_time)

        #* for logging in async sampler event
        commanded_position_ref[0] = start_pos

        commands = [
            {
                'actuator_id': actuator_id,
                'position': start_pos,
            }
        ]
        await kos.actuator.command_actuators(commands)

        await asyncio.sleep(1.0)
        await sampler
    
    elif wave_type == "chirp":
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

            #* for logging in async sampler event
            commanded_position_ref[0] = position
            
            commands = [
                {
                    'actuator_id': actuator_id,
                    'position': position,
                    'velocity': velocity,
                }
            ]
            
            await kos.actuator.command_actuators(commands)


    else:
        raise ValueError(f"Invalid wave type: {wave_type}")


    # Save collected data to JSON
    fldr_name = f"DATA/{datetime.now().strftime('%Y%m%d')}/{wave_type}"
    filename = f"{fldr_name}/{simorreal}_{actuator_id}_kp{kp}_kd{kd}.json"
    
    # Convert data to list of dictionaries for JSON serialization
    json_data = []
    for entry in DATA:
        json_data.append({
            "time_since_start": entry[0],
            "position": entry[1],
            "commanded_position": entry[2],
            "torque": entry[3]
        })
    
    # Create JSON structure with config as header and data points
    output = {
        "config": {
            "mode": simorreal,
            "wave_type": wave_type,
            "time_start": collected_at,
            "kp": kp,
            "kd": kd,
            "max_torque": max_torque,
            "step_size": step_size,
            "step_hold_time": step_hold_time,
            "step_count": step_count,
            "start_pos": start_pos,
            "actuator_id": actuator_id,
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


async def run_per(wave_type, sim, kos, joint_name, input_kp, input_kd, input_start_pos):
    TEST_CONFIGS = {
        "joint_name": joint_name,
        "simorreal": "sim" if sim else "real",
        "collected_at": datetime.now().strftime("%Y%m%d_%H%M%S"),

        "step_hold_time": 2.0, # seconds
        "step_count": 100,  # 1000
        "start_pos": input_start_pos,  # degrees

        "step_size": -10.0,       # degrees

        "start_freq": 0.2,
        "end_freq": 2.0,
        "chirp_duration": 7.0,

    }

    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    try:
        joint_name = TEST_CONFIGS["joint_name"]
        joint_metadata = metadata["joint_name_to_metadata"].get(joint_name)
        actuator_id = joint_metadata["id"]
        TEST_CONFIGS["max_torque"] = float(joint_metadata["soft_torque_limit"])
    
        #! Using the set_kp, set_kd, overriding the metadata
        TEST_CONFIGS["kp"] = float(input_kp)
        TEST_CONFIGS["kd"] = float(input_kd)
        
        actuator_type = joint_metadata.get("actuator_type")
        passive_params = metadata["actuator_type_passive_param"][actuator_type]
        TEST_CONFIGS["armature"] = float(passive_params["armature"])
        TEST_CONFIGS["frictionloss"] = float(passive_params["frictionloss"])
        TEST_CONFIGS["damping"] = float(passive_params["damping"])
            
        frc_range = passive_params["actuatorfrcrange"].split()
        TEST_CONFIGS["actuatorfrcrange"] = [float(frc_range[0]), float(frc_range[1])]

        #* For from standing
        state = await kos.actuator.get_actuators_state([actuator_id])
        TEST_CONFIGS["start_pos"] = float(state.states[0].position)

        print(TEST_CONFIGS["start_pos"], actuator_id)

    except Exception as e:
        logger.error(f"Metadata.json defined incorrectly")
        exit(1)

    await run_test(wave_type, kos, actuator_id, **TEST_CONFIGS)


async def go_to_zero(kos, sim):
    for id in [31, 32, 33, 34, 35, 41, 42, 43, 44, 45]:
        try:
            await kos.actuator.configure_actuator(actuator_id=id, kp=50.0, kd=5.0, torque_enabled=True)
        except Exception as e:
            print(f"Failed to configure actuator {id}")
    await asyncio.sleep(1)

    commands = [{'actuator_id': id, 'position': 0.0, 'velocity': 10.0} for id in [31, 32, 33, 34, 35, 41, 42, 43, 44, 45]]
    await kos.actuator.command_actuators(commands)

    if sim:
        await kos.sim.reset(
            pos={"x": 0.0, "y": 0.0, "z": 1.01},
            quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        )

async def go_to_stable_stand(kos, sim):
    # Load metadata to get kp and kd values
    with open('metadata.json', 'r') as f:
        local_metadata = json.load(f)
    
    
    # Map of actuator ids to their joint names
    id_to_joint_name = {}
    for joint_name, joint_data in local_metadata["joint_name_to_metadata"].items():
        id_to_joint_name[int(joint_data["id"])] = joint_name
    
    # Configure all actuators
    for id in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45]:
        try:
            joint_name = id_to_joint_name.get(id)
            if joint_name:
                joint_metadata = local_metadata["joint_name_to_metadata"][joint_name]
                kp = float(joint_metadata["kp"])
                kd = float(joint_metadata["kd"])
                max_torque = float(joint_metadata["soft_torque_limit"])
                await kos.actuator.configure_actuator(
                    actuator_id=id, 
                    kp=kp, 
                    kd=kd, 
                    max_torque=max_torque,
                    torque_enabled=True
                )
                logger.info(f"Configured actuator {id} with kp={kp}, kd={kd}, max_torque={max_torque}")
        except Exception as e:
            logger.error(f"Failed to configure actuator {id}: {e}")

    commands = [
        {'actuator_id': 11, 'position': 0.0, 'velocity': 10.0},        # dof_left_shoulder_pitch_03
        {'actuator_id': 12, 'position': 10.0, 'velocity': 10.0},       # dof_left_shoulder_roll_03
        {'actuator_id': 13, 'position': 0.0, 'velocity': 10.0},        # dof_left_shoulder_yaw_02
        {'actuator_id': 14, 'position': -90.0, 'velocity': 10.0},      # dof_left_elbow_02
        {'actuator_id': 15, 'position': 0.0, 'velocity': 10.0},        # dof_left_wrist_00
        {'actuator_id': 21, 'position': 0.0, 'velocity': 10.0},        # dof_right_shoulder_pitch_03
        {'actuator_id': 22, 'position': -10.0, 'velocity': 10.0},      # dof_right_shoulder_roll_03
        {'actuator_id': 23, 'position': 0.0, 'velocity': 10.0},        # dof_right_shoulder_yaw_02
        {'actuator_id': 24, 'position': 90.0, 'velocity': 10.0},       # dof_right_elbow_02
        {'actuator_id': 25, 'position': 0.0, 'velocity': 10.0},        # dof_right_wrist_00
        {'actuator_id': 31, 'position': 25.0, 'velocity': 10.0},       # dof_left_hip_pitch_04
        {'actuator_id': 32, 'position': 0.0, 'velocity': 10.0},        # dof_left_hip_roll_03
        {'actuator_id': 33, 'position': 0.0, 'velocity': 10.0},        # dof_left_hip_yaw_03
        {'actuator_id': 34, 'position': 50.0, 'velocity': 10.0},       # dof_left_knee_04
        {'actuator_id': 35, 'position': -25.0, 'velocity': 10.0},      # dof_left_ankle_02
        {'actuator_id': 41, 'position': -25.0, 'velocity': 10.0},      # dof_right_hip_pitch_04
        {'actuator_id': 42, 'position': 0.0, 'velocity': 10.0},        # dof_right_hip_roll_03
        {'actuator_id': 43, 'position': 0.0, 'velocity': 10.0},        # dof_right_hip_yaw_03
        {'actuator_id': 44, 'position': -50.0, 'velocity': 10.0},      # dof_right_knee_04
        {'actuator_id': 45, 'position': 25.0, 'velocity': 10.0},       # dof_right_ankle_02
    ]

    if sim:
        print("Resetting sim")
        await kos.sim.reset(pos={"x": 0.0, "y": 0.0, "z": 1.01}, quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})

    await kos.actuator.command_actuators(commands)




async def main(wave_type, sim):
    kos = pykos.KOS("0.0.0.0")
    
    joint_names = [
        "dof_right_hip_pitch_04",
        # "dof_right_hip_roll_03",
        # "dof_right_hip_yaw_03",
        # "dof_right_knee_04",
        # "dof_right_ankle_02",


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


    await go_to_stable_stand(kos, sim)
    await asyncio.sleep(5.0)

    for joint_name in joint_names:
        if joint_name == "dof_right_hip_pitch_04":
            kp = 150.0
            kd = 24.722
            start_pos = -25.0
        elif joint_name == "dof_right_hip_roll_03":
            kp = 200.0
            kd = 26.387
            start_pos = 0.0
        elif joint_name == "dof_right_hip_yaw_03":
            kp = 100.0
            kd = 3.419
            start_pos = 0.0
        elif joint_name == "dof_right_knee_04":
            kp = 150.0
            kd = 8.654
            start_pos = -50.0
        elif joint_name == "dof_right_ankle_02":
            kp = 40.0
            kd = [0.990]
            start_pos = 25.0
        else:
            raise ValueError(f"Invalid joint name: {joint_name}")

        # for kp in kp_list:
        #     for kd in kd_list:
        await run_per(wave_type, sim, kos, joint_name, kp, kd, start_pos)
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--input_type", type=str, choices=["step", "chirp"], default="step", help="Type of test to run: 'step' or 'wave'")
    args = parser.parse_args()

    asyncio.run(main(args.input_type, args.sim))
