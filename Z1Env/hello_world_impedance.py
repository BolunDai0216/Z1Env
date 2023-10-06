import argparse
import time

import numpy as np
import pinocchio as pin
import pybullet as p

from Z1Env import getDataPath
from Z1Env.z1_env import Z1Sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reachable", action="store_true", default=False)
    args = parser.parse_args()

    env = Z1Sim(render_mode="human")
    info = env.reset()

    # set desired end-effector position and orientation
    p_des = np.array([0.5, 0.0, 0.5])
    R_des = pin.rpy.rpyToMatrix(np.pi / 2, -np.pi / 4, 0.0)

    package_directory = getDataPath()
    urdf_search_path = package_directory + "/robots"
    p.setAdditionalSearchPath(urdf_search_path)

    target = p.loadURDF("target.urdf", useFixedBase=True)
    p.resetBasePositionAndOrientation(target, p_des.tolist(), [0, 0, 0, 1])

    if not args.reachable:
        block = p.loadURDF("block.urdf", useFixedBase=True)
        p.resetBasePositionAndOrientation(block, p_des.tolist(), [0, 0, 0, 1])

    for i in range(20000):
        # compute end-effector velocity
        v_EE = info["J_EE"] @ info["dq"][:, np.newaxis]

        # compute orientation error
        R_error = R_des @ info["R_EE"].T
        ori_error = pin.log3(R_error)

        # compute position error
        pos_error = p_des - info["P_EE"]

        # concatenate position and orientation error
        error = np.concatenate((pos_error, ori_error), axis=0)[:, np.newaxis]

        # set PD gains
        Kp = np.diag([20, 20, 20, 2, 2, 2])
        Kd = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])

        # set desired velocity to zero
        v_des = np.zeros((6, 1))

        # compute impendance force
        F = Kp @ error + Kd @ (v_des - v_EE)

        # compute joint torques from impedance force
        tau_impedance = info["J_EE"].T @ F

        # compute joint torques with gravity compensation
        tau = tau_impedance[:, 0] + info["G"]

        # Send joint commands to motor
        info = env.step(tau)
        time.sleep(1e-3)

        if i % 100 == 0:
            print(f"Error: {np.linalg.norm(error):.4f}")

    env.close()


if __name__ == "__main__":
    main()
