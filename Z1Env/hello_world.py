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
        q_des = np.array([0.0, 0.625, -0.424, 0.1, 0.1, 0.1, -0.1])
        dq_des = np.zeros(7)

        tau = 5 * (q_des - info["q"]) + 0.5 * (dq_des - info["dq"]) + info["G"]

        # print(tau)

        # Send joint commands to motor
        info = env.step(tau)

        # for i in range(p.getNumJoints(env.robotID)):
        #     joint_info = p.getJointState(env.robotID, i)
        #     reaction_forces = joint_info[2]
        #     print("Joint", i, "Reaction Forces:", reaction_forces)

        time.sleep(1e-3)

    env.close()


if __name__ == "__main__":
    main()
