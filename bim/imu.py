import argparse
import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime
from time import perf_counter, time as now

import colorlogging
import numpy as np
from pykos import KOS

logger = logging.getLogger(__name__)
colorlogging.configure()

from typing import List


async def sample_loop(
    kos: KOS, imu_kos: KOS, DATA: List[dict], duration: float, actuator_id: int
):
    sampling_rate = 100.0  # Hz
    sampling_period = 1.0 / sampling_rate
    end_time = time.perf_counter() + duration + 2  # +2.0 for extra data at end
    
    while time.perf_counter() < end_time:
        loop_start = time.perf_counter()
        
        # Gather all sensor data
        state = await kos.actuator.get_actuators_state([actuator_id])
        raw_data = await imu_kos.imu.get_imu_values()
        quat = await imu_kos.imu.get_quaternion()
        current_time = time.perf_counter()

        next_entry = {
            "time": current_time,
            "position": state.states[0].position,
            "accel_xyz": (raw_data.accel_x, raw_data.accel_y, raw_data.accel_z),
            "gyro_xyz": (raw_data.gyro_x, raw_data.gyro_y, raw_data.gyro_z),
            "mag_xyz": (raw_data.mag_x, raw_data.mag_y, raw_data.mag_z),
            "quat_xyzw": (quat.x, quat.y, quat.z, quat.w),
        }
        DATA.append(next_entry)
        
        elapsed = time.perf_counter() - loop_start
        sleep_time = sampling_period - elapsed
        
        if sleep_time < 0:
            logger.warning(f"Sampling loop overran by {-sleep_time:.6f} seconds")
        else:
            await asyncio.sleep(sleep_time)


def save_data(
    DATA: List[dict],
    input_type: str,
    sim: bool,
    title: str,
    actuator_id: int,
    amplitude: float,
    duration: float,
) -> None:

    simorreal = "sim" if sim else "real"

    collect_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_data = {
        "config": {
            "input_type": input_type,
            "collected_on": collect_time,
            "mode": simorreal,
            "actuator_id": actuator_id,
            "amplitude": amplitude,
            "duration": duration,
            "title": title,
        },
        "data": DATA,
    }

    fldr_name = f"DATA/{collect_time.split('_')[0]}/{input_type}"
    os.makedirs(fldr_name, exist_ok=True)
    with open(f"{fldr_name}/{simorreal}_{collect_time}{title}.json", "w") as f:
        json.dump(final_data, f, indent=2)
        logger.info(f"Saved data to {fldr_name}/{simorreal}_{collect_time}{title}.json")


async def run_test(
    kos: KOS,
    imu_kos: KOS,
    DATA: List[dict],
    title: str,
    actuator_id: int,
    input_type: str,
    sim: bool,
    amplitude: float,
    duration: float,
    servo_only: bool,
):
    test_start_time = time.perf_counter()

    if sim:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=150.0,
            kd=10.0,
            acceleration=200.0,
            torque_enabled=True,
        )

    else:
        await kos.actuator.configure_actuator(
            actuator_id=actuator_id,
            kp=30.0,
            kd=10.0,
            acceleration=200.0,
            torque_enabled=True,
        )

    await kos.actuator.command_actuators(
        [
            {
                "actuator_id": actuator_id,
                "position": 0,
            }
        ]
    )

    logger.info("Actuator configured")

    if not servo_only:
        sampler = asyncio.create_task(
            sample_loop(kos, imu_kos, DATA, duration, actuator_id)
        )

    if input_type == "step":

        await asyncio.sleep(1.0)

        commands = [
            {
                "actuator_id": actuator_id,
                "position": amplitude,
                "velocity": 0.0,
            }
        ]
        await kos.actuator.command_actuators(commands)
        await asyncio.sleep(duration - 2.0)

        commands = [
            {
                "actuator_id": actuator_id,
                "position": 0.0,
            }
        ]
        await kos.actuator.command_actuators(commands)

        await asyncio.sleep(1.0)

    elif input_type == "chirp":
        end_freq = 1.5
        start_freq = 0.2

        command_freq = 50 #hz

        k = (end_freq - start_freq) / duration  # Rate of frequency change
        f0 = start_freq

        current_time = time.perf_counter() - test_start_time
        while current_time < duration:
            loop_t0 = time.perf_counter()
            current_time = time.perf_counter() - test_start_time

            phase = (
                2.0
                * math.pi
                * (f0 * current_time + 0.5 * k * current_time * current_time)
            )

            # Calculate instantaneous frequency and angular velocity
            freq = f0 + k * current_time
            omega = 2.0 * math.pi * freq

            # Calculate position and velocity
            wave_amplitude = amplitude / 2.0
            position = wave_amplitude * np.sin(phase)
            velocity = wave_amplitude * omega * np.cos(phase)

            commands = [
                {
                    "actuator_id": actuator_id,
                    "position": position,
                    "velocity": velocity,
                }
            ]

            await kos.actuator.command_actuators(commands)
            if 1.0 / command_freq - (time.perf_counter() - loop_t0) > 0.0:
                await asyncio.sleep(1.0 / command_freq - (time.perf_counter() - loop_t0))
            else:
                logger.warning(f"Chirp loop overran by {time.perf_counter() - loop_t0 - 1.0 / command_freq:.6f} seconds")
    elif input_type == "static":
        pass
    else:
        raise ValueError(f"Invalid input type: {input_type}")

    if not servo_only:
        await sampler

    CONFIG = {
        "actuator_id": 1,
        "input_type": input_type,
        "sim": sim,
        "amplitude": amplitude,
        "duration": duration,
        "title": title,
    }
    if not servo_only:
        save_data(DATA, **CONFIG)


async def main(input_type: str, sim: bool, DATA: List[dict], title: str, servo_only: bool):

    CONFIG = {
        "actuator_id": 1,
        "input_type": input_type,
        "sim": sim,
        "amplitude": 30.0,
        "duration": 6.0,
        "title": title,
    }

    if sim:
        kos = KOS("100.101.101.48")
        imu_kos = kos

        #! Temp fix, MJCF motor orientation is wrong
        CONFIG["amplitude"] = -1*CONFIG["amplitude"]
    else:
        kos = KOS("0.0.0.0", "3001")
        if servo_only:
            imu_kos = KOS("0.0.0.0", "3001") #*Won't be used
        else:
            imu_kos = KOS("0.0.0.0")
        CONFIG["actuator_id"] = 14

    try:
        await run_test(kos, imu_kos, DATA, **CONFIG, servo_only=servo_only)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
        if not servo_only:
            save_data(DATA, **CONFIG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    parser.add_argument(
        "--input_type", type=str, choices=["step", "chirp", "static"], default="step"
    )
    parser.add_argument(
        "--title", type=str, default=""
    )
    parser.add_argument(
        "--servo_only", action="store_true"
    )
    args = parser.parse_args()

    DATA: List[dict] = []
    asyncio.run(main(args.input_type, args.sim, DATA, args.title, args.servo_only))
