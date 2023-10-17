import time

import numpy as np

from Z1Env.z1_env import Z1Sim
from Z1Env.spline_gen import SplineGenerator


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    dt = 5e-4
    q_term = np.array([[0, 0.671, -0.500, -0.144, 0.0, 0.0, 0]])

    Kp = np.diag([20, 20, 20, 2, 2, 2, 2])
    Kd = np.diag([0.02, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2])
    T = 2

    spline = SplineGenerator(info["q"].reshape((1, 7)), q_term, T)

    for i in range(20000):
        t = i * dt

        if t >= T:
            t = T

        q = spline.get_q(t)[:, 0]
        dq = spline.get_dq(t)[:, 0]

        tau = Kp @ (q - info["q"]) + Kd @ (dq - info["dq"]) + info["G"]

        info = env.step(tau)

        time.sleep(dt)


if __name__ == "__main__":
    main()
