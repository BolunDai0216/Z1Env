import time

import numpy as np
from ndcurves import polynomial

from Z1Env import Z1Sim


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    dt = 5e-4
    T = 2.0

    q_term = np.array([0, 0.671, -0.500, -0.144, 0.0, 0.0, 0])
    Kp = np.diag([20, 20, 20, 2, 2, 2, 2])
    Kd = np.diag([0.02, 0.02, 0.02, 0.2, 0.2, 0.2, 0.2])

    curve = polynomial.MinimumJerk(info["q"], q_term, 0.0, T)

    for i in range(20000):
        t = np.clip(i * dt, 0.0, T)
        q = curve(t)
        dq = curve.derivate(t, 1)

        tau = Kp @ (q - info["q"]) + Kd @ (dq - info["dq"]) + info["G"]
        info = env.step(tau)

        time.sleep(dt)


if __name__ == "__main__":
    main()
