import copy
from typing import Optional

import numpy as np
import pinocchio as pin
import pybullet as p
import pybullet_data
from gymnasium import Env, spaces
from pinocchio.robot_wrapper import RobotWrapper

from Z1Env import getDataPath


def main():
    p.connect(p.GUI)

    p.setGravity(0, 0, 0.0)
    p.setTimeStep(1 / 240)

    # Load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # Load Unitree Z1 Arm
    package_directory = getDataPath()
    urdf_search_path = package_directory + "/robots"
    p.setAdditionalSearchPath(urdf_search_path)
    robotID = p.loadURDF("z1.urdf", useFixedBase=True)

    # Get number of joints
    n_j = p.getNumJoints(robotID)

    debug_sliders = []
    joint_ids = []

    default_joint_angles = [0.0] * 7
    counter = 0

    for i in range(n_j):
        # get info of each joint
        _joint_infos = p.getJointInfo(robotID, i)

        if _joint_infos[2] != p.JOINT_FIXED:
            # Add a debug slider for all non-fixed joints
            debug_sliders.append(
                p.addUserDebugParameter(
                    _joint_infos[1].decode("UTF-8"),  # Joint Name
                    _joint_infos[8],  # Lower Joint Limit
                    _joint_infos[9],  # Upper Joint Limit
                    default_joint_angles[counter],  # Default Joint Angle
                )
            )

            # Save the non-fixed joint IDs
            joint_ids.append(_joint_infos[0])
            counter += 1

    while True:
        for slider_id, joint_id in zip(debug_sliders, joint_ids):
            # Get joint angle from debug slider
            try:
                _joint_angle = p.readUserDebugParameter(slider_id)
            except:
                # Sometimes it fails to read the debug slider
                continue

            # Apply joint angle to robot
            p.resetJointState(robotID, joint_id, _joint_angle)

        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
