import time

import hppfcl
import numpy as np
import pinocchio as pin

from Z1Env import Z1Sim


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    link2_id = env.robot.model.getFrameId("link2")
    link6_id = env.robot.model.getFrameId("link6")

    breakpoint()

    for i in range(10000):
        q_des = np.array([0.0, 0.625, -0.424, 0.1, 0.1, 0.1, -0.1])
        dq_des = np.zeros(7)
        tau = 5 * (q_des - info["q"]) + 0.5 * (dq_des - info["dq"]) + info["G"]

        # Send joint commands to motor
        info = env.step(tau)
        time.sleep(1e-3)

    env.close()


if __name__ == "__main__":
    main()
