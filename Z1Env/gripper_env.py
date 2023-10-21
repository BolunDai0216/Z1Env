import time
from typing import Optional

from gymnasium import Env
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import pybullet as p
import pybullet_data

from Z1Env import getDataPath


class GripperSim(Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self, render_mode: Optional[str] = None, record_path=None, crude_model=False
    ):
        if render_mode == "human":
            self.client = p.connect(p.GUI)
            # Improves rendering performance on M1 Macs
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)

        self.record_path = record_path

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(5e-4)

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Load Unitree Z1 Arm
        package_directory = getDataPath()
        robot_URDF = package_directory + "/robots/gripper.urdf"
        urdf_search_path = package_directory + "/robots"
        p.setAdditionalSearchPath(urdf_search_path)
        self.robotID = p.loadURDF("gripper.urdf", useFixedBase=True)

        # Build pin_robot
        self.robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

        # Get number of joints
        self.n_j = p.getNumJoints(self.robotID)
        self.active_joint_ids = []

        for i in range(self.n_j):
            # get info of each joint
            _joint_infos = p.getJointInfo(self.robotID, i)

            if _joint_infos[2] != p.JOINT_FIXED:
                # Save the non-fixed joint IDs
                self.active_joint_ids.append(_joint_infos[0])

        self.n_j_active = len(self.active_joint_ids)

        p.resetBasePositionAndOrientation(self.robotID, [0.0, 0.0, 0.0], [0, 0, 0, 1])

        # End-effector frame id
        self.EE_FRAME_ID = self.robot.model.getFrameId("tcp_link")
        self.BASE_FRAME_ID = self.robot.model.getFrameId("link6")

        # Compute the transformation matrix between the base and the tcp_link
        self.robot.framesForwardKinematics(np.array([0.0]))
        self.T_gripper = self.robot.data.oMf[self.EE_FRAME_ID].homogeneous
        self.T_gripper_inv = np.linalg.inv(self.T_gripper)

        # Get frame ID for grasp target
        self.jacobian_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        cameraDistance=0.5,
        cameraYaw=66.4,
        cameraPitch=-16.2,
        lookat=[0.0, 0.0, 0.0]
    ):
        super().reset(seed=seed)

        p.resetJointState(self.robotID, 1, 0.0, 0.0)

        if self.record_path is not None:
            p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, lookat)

            self.loggingId = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, self.record_path
            )

        pos = np.zeros((3, 1))
        rot_mat = np.eye(3)
        base_pos, base_quat = self.get_pose(pos, rot_mat)

        p.resetBasePositionAndOrientation(
            self.robotID, base_pos.tolist(), base_quat.tolist()
        )

    def step(self, action):
        pos = action["pos"][:, np.newaxis]
        rot_mat = action["rot_mat"]
        base_pos, base_quat = self.get_pose(pos, rot_mat)

        p.resetBasePositionAndOrientation(
            self.robotID, base_pos.tolist(), base_quat.tolist()
        )

    def close(self):
        if self.record_path is not None:
            p.stopStateLogging(self.loggingId)
        p.disconnect()

    def get_pose(self, pos, rot_mat):
        _T_gripper = pin.SE3(rot_mat, pos)
        T_base = pin.SE3(_T_gripper @ self.T_gripper_inv)

        base_pos = T_base.translation
        base_quat = pin.Quaternion(T_base.rotation).coeffs()

        return base_pos, base_quat


def main():
    env = GripperSim(render_mode="human")
    _ = env.reset()

    for i in range(100000):
        t = 1 * i / 100

        action = {
            "pos": np.array([0.0, 0.0, 1.0 + 0.5 * np.sin(t)]),
            "rot_mat": np.eye(3),
        }

        _ = env.step(action)
        time.sleep(0.01)


if __name__ == "__main__":
    main()
