import time
from IPython import display

import numpy as np
import matplotlib.pyplot as plt

import sys
# on arm installed as robotpck source pinocchio there
sys.path.append('/usr/local/lib/python3/dist-packages')

from Z1Env import Z1Sim


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    for i in range(10000):
        q_des = np.array([0.0, 0.625, -0.424, 0.1, 0.1, 0.1, -0.1])
        dq_des = np.zeros(7)
        tau = 5 * (q_des - info["q"]) + 0.5 * (dq_des - info["dq"]) + info["G"]

        # Send joint commands to motor
        # plt.imshow(env.render(render_mode="rgb_array"))
        info = env.step(tau)
        #plt.imshow(env.render())
        time.sleep(1e-3)

    env.close()


if __name__ == "__main__":
    main()
