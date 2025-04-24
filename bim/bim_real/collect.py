import asyncio
import math
import time
import csv
from datetime import datetime
import pickle
import os
from pykos import KOS
import imu
import argparse
import colorlogging
import logging
from scipy.spatial.transform import Rotation
import numpy as np

logger = logging.getLogger(__name__)
colorlogging.configure()


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




def save_data(data):
    """Save IMU data to a pickle file.
    
    Args:
        data: IMUData object containing all the collected data
    """
    
    # Determine base directory relative to this file and build nested folders: data/YYYYMMDD/imu/{amplitude}_{frequency}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    date_str = datetime.now().strftime("%Y%m%d")
    file_prefix = f"amp{data.amplitude}_freq{data.frequency}"
    dir_path = os.path.join(base_dir, "../data", f"{date_str}_real")
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

async def per_loop(reader, kos, data):
    if output_data := reader.get_data():
        if 'quaternion' in output_data and output_data['quaternion'] is not None:
            data.quat_w.append(output_data['quaternion'].w)
            data.quat_x.append(output_data['quaternion'].x)
            data.quat_y.append(output_data['quaternion'].y)
            data.quat_z.append(output_data['quaternion'].z)

            r = Rotation.from_quat([data.quat_w[-1], data.quat_x[-1], data.quat_y[-1], data.quat_z[-1]], scalar_first=True)
            proj_grav_world = r.apply(np.array([0.0, 0.0, 1.0]), inverse=True)
            projected_gravity = proj_grav_world

            data.proj_x.append(projected_gravity[0])
            data.proj_y.append(projected_gravity[1])
            data.proj_z.append(projected_gravity[2])

            euler_from_quat = r.as_euler('xyz', degrees=True)
            data.euler_x_quat.append(euler_from_quat[0])
            data.euler_y_quat.append(euler_from_quat[1])
            data.euler_z_quat.append(euler_from_quat[2])
        else:
            raise ValueError("Quaternion data not found")

        if 'euler' in output_data and output_data['euler'] is not None:
            data.euler_x.append(output_data['euler'].x * 180 / math.pi)
            data.euler_y.append(output_data['euler'].y * 180 / math.pi)
            data.euler_z.append(output_data['euler'].z * 180 / math.pi)
        else:
            raise ValueError("Euler data not found")

        if 'accelerometer' in output_data and output_data['accelerometer'] is not None:
            data.acc_x.append(output_data['accelerometer'].x)
            data.acc_y.append(output_data['accelerometer'].y)
            data.acc_z.append(output_data['accelerometer'].z)
        else:
            raise ValueError("Accelerometer data not found")
        
        if 'gyroscope' in output_data and output_data['gyroscope'] is not None:
            data.gyro_x.append(output_data['gyroscope'].x)
            data.gyro_y.append(output_data['gyroscope'].y)
            data.gyro_z.append(output_data['gyroscope'].z)
        else:
            raise ValueError("Gyroscope data not found")


async def main():
    parser = argparse.ArgumentParser(description="BIM real experiment.")
    parser.add_argument(
        "--device", type=str, default="/dev/ttyUSB0", help="Serial device path for the IMU (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baudrate", type=int, default=230400, help="Serial baud rate for the IMU (default: 230400)"
    )
    parser.add_argument(
        "--kos-ip", type=str, default="127.0.0.1", help="KOS server IP address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Duration of the experiment (default: 10.0)"
    )
    parser.add_argument(
        "--amplitude", type=float, default=50.0, help="Amplitude of the sine wave (default: 1.0)"
    )
    parser.add_argument(
        "--frequency", type=float, default=0.5, help="Frequency of the sine wave (default: 1.0)"
    )
    parser.add_argument(
        "--data_freq", type=int, default=100, help="Data frequency in hz (default: 100)"
    )
    args = parser.parse_args()


    try:
        reader = imu.create_hiwonder(args.device, args.baudrate)
        print("IMU connection successful.")
    except Exception as e:
        print(f"Error connecting to IMU: {e}")
        print("Please check the device path, permissions, and baudrate.")
        return
    
    try:
        kos = KOS(args.kos_ip)
        print("KOS connection successful.")
    except Exception as e:
        print(f"Error connecting to KOS: {e}")
        print("Please check the KOS server is running.")
        return


    await kos.actuator.configure_actuator(
        actuator_id=100,
        torque_enabled=True,
        acceleration=1000,
        kp=20,
        kd=5
    )

    await asyncio.sleep(1)

    await kos.actuator.command_actuators([{
        'actuator_id': 100,
        'position': 0.0,
    }])

    await asyncio.sleep(5.0)


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

    start_time = time.time()
    end_time = start_time + args.duration

    per_loop_time = 1.0 / args.data_freq


    try:
        while time.time() < end_time:
            cycle_start_time = time.time()

            elapsed = time.time() - start_time
            position = args.amplitude * math.sin(2 * math.pi * args.frequency * elapsed)

            await kos.actuator.command_actuators([{
                'actuator_id': 100,
                'position': position,
            }])

            await per_loop(reader, kos, data)

            if time.time() - cycle_start_time > per_loop_time:
                logger.warning(f"Loop overrun by {time.time() - cycle_start_time - per_loop_time:.2f} seconds")
                data.loop_freq.append(1.0 / (time.time() - cycle_start_time))
            else:
                data.loop_freq.append(1.0 / per_loop_time)
                await asyncio.sleep(per_loop_time - (time.time() - cycle_start_time))

            data.timestamp.append(time.time())

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt")
        save_data(data)
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        save_data(data)

                


if __name__ == "__main__":
    asyncio.run(main())
