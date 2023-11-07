import time

import numpy as np
import pybullet as p

from Z1Env import OOBDetection, Z1Sim, getDataPath


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    package_directory = getDataPath()
    urdf_search_path = package_directory + "/robots"
    p.setAdditionalSearchPath(urdf_search_path)

    dimensions = [0.6, 0.5, 1]
    oob_detector = OOBDetection(dimensions, env)

    base_pos, base_quat = p.getBasePositionAndOrientation(env.robotID)
    oob_detector.visualize_box(base_pos, base_quat)

    for i in range(10000):
        out_of_bound = oob_detector.check_out_of_bound()

        if i % 100 == 0:
            if out_of_bound:
                p.changeVisualShape(
                    oob_detector.box, -1, rgbaColor=[1.0, 0.0, 0.0, 0.3]
                )
            else:
                p.changeVisualShape(
                    oob_detector.box, -1, rgbaColor=[0.0, 1.0, 0.0, 0.3]
                )

        # compute joint torques
        q_des = np.array([0.0, 1.6, -1.5, 0.1, 0.1, 0.1, -0.1])
        dq_des = np.zeros(7)
        tau = 5 * (q_des - info["q"]) + 0.5 * (dq_des - info["dq"]) + info["G"]

        # send joint commands to motor
        info = env.step(tau)
        time.sleep(1e-3)

    env.close()


if __name__ == "__main__":
    main()
