import time

import hppfcl
import numpy as np
import pinocchio as pin
import pybullet as p

from Z1Env import Z1Sim


class SelfCollisionChecker:
    def __init__(self):
        # create hppfcl instances
        self.link2_cylinder_hppfcl = hppfcl.Cylinder(0.05, 0.5)
        self.link6_cylinder_hppfcl = hppfcl.Cylinder(0.05, 0.2)

        # define constant offsets
        self.rotation_offset = pin.SE3(
            pin.rpy.rpyToMatrix(0.0, np.pi / 2, 0.0), np.zeros(3)
        )
        self.link2_offset = (
            pin.SE3(np.eye(3), np.array([-0.15, 0.0, 0.0])) * self.rotation_offset
        )
        self.link6_offset = (
            pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0])) * self.rotation_offset
        )

    def check_collision(self, link2_pose, link6_pose, return_pose=False):
        link2_cylinder_pose = link2_pose * self.link2_offset
        link6_cylinder_pose = link6_pose * self.link6_offset

        req = hppfcl.CollisionRequest()
        res = hppfcl.CollisionResult()

        T_link2 = hppfcl.Transform3f(
            link2_cylinder_pose.rotation, link2_cylinder_pose.translation
        )
        T_link6 = hppfcl.Transform3f(
            link6_cylinder_pose.rotation, link6_cylinder_pose.translation
        )

        in_collision = hppfcl.collide(
            self.link2_cylinder_hppfcl,
            T_link2,
            self.link6_cylinder_hppfcl,
            T_link6,
            req,
            res,
        )

        if return_pose:
            poses = {
                "link2": link2_cylinder_pose,
                "link6": link6_cylinder_pose,
            }
        else:
            poses = {}

        return in_collision, poses


def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    link2_id = env.robot.model.getFrameId("link2")
    link6_id = env.robot.model.getFrameId("link6")

    link2_cylinder = p.loadURDF("link2_cylinder.urdf", useFixedBase=True)
    link6_cylinder = p.loadURDF("link6_cylinder.urdf", useFixedBase=True)

    self_collision_checker = SelfCollisionChecker()

    for i in range(10000):
        q_des = np.array([0.0, 0.625, -0.424, 0.1, 0.1, 0.1, -0.1])
        dq_des = np.zeros(7)
        tau = 5 * (q_des - info["q"]) + 0.5 * (dq_des - info["dq"]) + info["G"]

        link2_pose = env.robot.data.oMf[link2_id]
        link6_pose = env.robot.data.oMf[link6_id]

        in_collision, poses = self_collision_checker.check_collision(
            link2_pose, link6_pose, return_pose=True
        )

        link2_cylinder_pose = poses["link2"]
        link6_cylinder_pose = poses["link6"]

        p.resetBasePositionAndOrientation(
            link2_cylinder,
            link2_cylinder_pose.translation.tolist(),
            pin.Quaternion(link2_cylinder_pose.rotation).coeffs().tolist(),
        )
        p.resetBasePositionAndOrientation(
            link6_cylinder,
            link6_cylinder_pose.translation.tolist(),
            pin.Quaternion(link6_cylinder_pose.rotation).coeffs().tolist(),
        )

        if i % 10 == 0:
            if in_collision:
                p.changeVisualShape(link2_cylinder, -1, rgbaColor=[1.0, 0.0, 0.0, 0.3])
                p.changeVisualShape(link6_cylinder, -1, rgbaColor=[1.0, 0.0, 0.0, 0.3])
            else:
                p.changeVisualShape(link2_cylinder, -1, rgbaColor=[0.0, 1.0, 0.0, 0.3])
                p.changeVisualShape(link6_cylinder, -1, rgbaColor=[0.0, 1.0, 0.0, 0.3])

        # Send joint commands to motor
        info = env.step(tau)
        time.sleep(1e-3)

    env.close()


if __name__ == "__main__":
    main()
