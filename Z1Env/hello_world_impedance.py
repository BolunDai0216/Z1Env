import time

import numpy as np
import pybullet as p

from Z1Env import getDataPath
from Z1Env.z1_env import Z1Sim


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    p_des = np.array([0.5, 0.0, 0.5])

    package_directory = getDataPath()
    urdf_search_path = package_directory + "/robots"
    p.setAdditionalSearchPath(urdf_search_path)

    target = p.loadURDF("target.urdf", useFixedBase=True)
    block = p.loadURDF("block.urdf", useFixedBase=True)

    p.resetBasePositionAndOrientation(target, p_des.tolist(), [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(block, p_des.tolist(), [0, 0, 0, 1])

    for i in range(10000):
        Jp_EE = info["J_EE"][:3, :]
        v_EE = Jp_EE @ info["dq"][:, np.newaxis]

        F = 20 * (p_des - info["P_EE"])[:, np.newaxis] - 0.1 * v_EE
        tau_impedance = Jp_EE.T @ F

        tau = tau_impedance[:, 0] + info["G"]

        # Send joint commands to motor
        info = env.step(tau)
        time.sleep(1e-3)

    env.close()


if __name__ == "__main__":
    main()
