#!/usr/bin/env python3
import os
import mujoco
from mujoco.viewer import launch_passive
import time

def main():
    # build path to the MJCF
    mjcf_path = os.path.join(os.path.dirname(__file__), 'robot.suspended.mjcf')
    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(f"Could not find MJCF at {mjcf_path}")

    # load model and data
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    # reset data (zero qpos, qvel, etc.)
    mujoco.mj_resetData(model, data)
    
    # Set simulation parameters
    model.opt.timestep = 0.001  # 0.001 = 1000hz
    
    # Optional: Disable gravity if needed
    # model.opt.gravity[2] = 0

    # launch the passive simulator with time synchronization
    with launch_passive(model, data) as viewer:
        target_time = time.time()
        sim_time = 0.0


        actuator_names = [
            "dof_right_hip_pitch_04_ctrl",
            "dof_right_hip_roll_03_ctrl",
            "dof_right_hip_yaw_03_ctrl",
            "dof_right_knee_04_ctrl",
            "dof_right_ankle_02_ctrl",
        ]

        for name in actuator_names:
            # Get the integer ID of this actuator
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            
            # Read b2 = model.actuator_biasprm[act_id, 2]  (this is –kv)
            b2 = model.actuator_biasprm[act_id, 2]
            kd = -b2
            
            print(f"{name:30s}  kd = {kd:.3f}")

        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep
            viewer.sync()


            # Synchronize with real time
            target_time += model.opt.timestep
            current_time = time.time()
            if target_time - current_time > 0:
                time.sleep(target_time - current_time)

if __name__ == "__main__":
    main()
