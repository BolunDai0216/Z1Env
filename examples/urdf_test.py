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
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

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

    while True:
        p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()
