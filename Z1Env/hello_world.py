import time

import numpy as np
import numpy.linalg as LA
import pybullet as p
from scipy.spatial.transform import Rotation as R

from Z1Env.z1_env import Z1Sim


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    for i in range(10000):
        tau = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Set control for the gripper to zero
        # tau[-1] = 0.0

        # Send joint commands to motor
        info = env.step(tau)

        for i in range(p.getNumJoints(env.robotID)):
            joint_info = p.getJointState(env.robotID, i)
            reaction_forces = joint_info[3]
            print("Joint", i, "Reaction Forces:", reaction_forces)

        # breakpoint()

        time.sleep(1 / 240)

    env.close()


if __name__ == "__main__":
    main()
