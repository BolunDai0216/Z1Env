import time

import numpy as np
import pinocchio as pin
import pybullet as p

from Z1Env import getDataPath
from Z1Env.z1_env import Z1Sim


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    # set desired end-effector position and orientation
    p_des = np.array([0.5, 0.0, 0.5])
    R_des = pin.rpy.rpyToMatrix(0.0, 0.0, 0.0)

    package_directory = getDataPath()
    urdf_search_path = package_directory + "/robots"
    p.setAdditionalSearchPath(urdf_search_path)

    button = p.loadURDF("button.urdf", useFixedBase=True)
    p.resetBasePositionAndOrientation(
        button, p_des.tolist(), [0, np.sin(-np.pi / 4), 0, np.cos(-np.pi / 4)]
    )

    p.setJointMotorControl2(button, 0, controlMode=p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(
        button,
        0,
        p.POSITION_CONTROL,
        targetPosition=0.02,
        force=0.1,
        maxVelocity=100,
        positionGain=0.01,
        velocityGain=0.0,
    )

    for i in range(20000):
        # compute end-effector velocity
        v_EE = info["J_EE"] @ info["dq"][:, np.newaxis]

        # compute orientation error in the world frame
        R_error = R_des @ info["R_EE"].T
        ori_error = pin.log3(R_error)

        # compute position error
        pos_error = p_des - info["P_EE"]

        # concatenate position and orientation error
        error = np.concatenate((pos_error, ori_error), axis=0)[:, np.newaxis]
        pos_error = np.linalg.norm(error[:3])

        # set PD gains
        if pos_error >= 0.025:
            Kp = np.diag([10, 10, 10, 2, 2, 2])
        else:
            Kp_pos = 1.0 / pos_error
            Kp = np.diag([Kp_pos, Kp_pos, Kp_pos, 2, 2, 2])

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
            print(
                "\033[1m"
                + f"\r> Normalized Error: {np.linalg.norm(error):.4f}"
                + "\033[0m",
                end="",
                flush=True,
            )

    env.close()


if __name__ == "__main__":
    main()
