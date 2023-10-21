import time

from ndcurves import SE3Curve
import numpy as np
import pinocchio as pin
import pybullet as p

from Z1Env import DiffIK, Z1Sim


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    T_init = pin.SE3(info["R_EE"], info["P_EE"])
    R_end = pin.rpy.rpyToMatrix(0.0, 0.0, 0.0)
    p_end = np.array([[0.5], [0.0], [0.3]])
    T_end = pin.SE3(R_end, p_end)
    t_init = 0.0
    t_end = 5.0
    curve = SE3Curve(T_init, T_end, t_init, t_end)

    target = p.loadURDF("target.urdf", useFixedBase=True)
    p.resetBasePositionAndOrientation(target, p_end[:, 0].tolist(), [0, 0, 0, 1])

    controller = DiffIK()

    for i in range(100000):
        t = i * 1e-3

        T = pin.SE3(info["R_EE"], info["P_EE"])

        if t <= t_end:
            V = pin.log6(T.inverse().homogeneous @ curve(t)).vector
            V_des = 1.0 * V + curve.derivate(t, 1)
        else:
            V = pin.log6(T.inverse().homogeneous @ T_end).vector
            V_des = 1.0 * V + np.zeros((6,))

        dq_des = controller(V_des, info["q"], info["J_EE"])
        tau = 1.0 * (dq_des - info["dq"]) + info["G"] - 0.1 * info["dq"]

        info = env.step(tau)
        time.sleep(1e-4)

    env.close()


if __name__ == "__main__":
    main()
