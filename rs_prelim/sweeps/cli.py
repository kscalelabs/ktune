import argparse
import xml.etree.ElementTree as ET
import os
import json

user_root = "/Users/aaronxie"
mjcf_path = f"{user_root}/.kscale/robots/kbot-v2/robot/robot.suspended.mjcf"
script_metadata_path = f"{user_root}/Documents/Code_Local/KBOT/ktune/rs_prelim/metadata.json"

def update_mjcf_parameters(frictionloss, damping):
    # Parse the MJCF file
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    
    # Find all motor default classes and update parameters
    motor_classes = ["motor_00", "motor_02", "motor_03", "motor_04"]
    updates_made = 0
    
    for motor_class in motor_classes:
        # Find the joint elements within each motor class
        xpath = f".//default[@class='{motor_class}']/joint"
        joint_elements = root.findall(xpath)
        
        for joint in joint_elements:
            # Update the frictionloss and damping attributes
            joint.set("frictionloss", str(frictionloss))
            joint.set("damping", str(damping))
            updates_made += 1
    
    # Save the updated file
    tree.write(mjcf_path)
    print(f"Updated {updates_made} motor joints with frictionloss={frictionloss}, damping={damping}")
    print(f"Modified file: {mjcf_path}")

def update_metadata(frictionloss, damping):
    with open(script_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Update all motor types in the actuator_type_passive_param section
    motor_types = ["robstride_00", "robstride_02", "robstride_03", "robstride_04"]
    for motor_type in motor_types:
        if motor_type in metadata.get("actuator_type_passive_param", {}):
            metadata["actuator_type_passive_param"][motor_type]["frictionloss"] = str(frictionloss)
            metadata["actuator_type_passive_param"][motor_type]["damping"] = str(damping)
    
    with open(script_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated metadata with frictionloss={frictionloss}, damping={damping}")
    print(f"Modified file: {script_metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Update friction loss and damping in MJCF file')
    parser.add_argument('--frictionloss', type=float, required=True, help='Friction loss value for all motors')
    parser.add_argument('--damping', type=float, required=True, help='Damping value for all motors')
    
    args = parser.parse_args()
    
    # Check if the MJCF file exists
    if not os.path.exists(mjcf_path):
        print(f"Error: MJCF file not found at {mjcf_path}")
        return
    
    update_mjcf_parameters(args.frictionloss, args.damping)
    update_metadata(args.frictionloss, args.damping)

if __name__ == "__main__":
    main()




