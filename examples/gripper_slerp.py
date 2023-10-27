import time

import numpy as np
import pinocchio as pin
from ndcurves import SE3Curve

from Z1Env import GripperSim


def main():
    env = GripperSim(render_mode="human")
    env.reset()

    T_init = pin.SE3.Identity()
    R_end = pin.rpy.rpyToMatrix(0.0, np.pi / 2, np.pi / 2)
    p_end = np.array([[1.0], [1.0], [1.0]])
    T_end = pin.SE3(R_end, p_end)
    t_init = 0.0
    t_end = 9.0
    curve = SE3Curve(T_init, T_end, t_init, t_end)

    for i in range(100000):
        t = np.clip(1 * i / 1000, 0, t_end)
        T = curve(t)

        action = {
            "pos": T[:3, -1],
            "rot_mat": T[:3, :3],
        }

        env.step(action)
        time.sleep(1e-4)

    env.close()


if __name__ == "__main__":
    main()
