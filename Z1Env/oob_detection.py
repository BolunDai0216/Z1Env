import numpy as np
import pybullet as p


class OOBDetection:
    def __init__(self, box_size, env):
        """
        Class for out-of-boundary detection:

        Saves the dimension as length, width, and height of the box
        Fetches the Frame ID of each link mentioned in the robots/z1.urdf
        """
        self.env = env
        self.x_size = box_size[0]
        self.y_size = box_size[1]
        self.z_size = box_size[2]

        self.link1_ID = env.robot.model.getFrameId("link1")
        self.link2_ID = env.robot.model.getFrameId("link2")
        self.link3_ID = env.robot.model.getFrameId("link3")
        self.link4_ID = env.robot.model.getFrameId("link4")
        self.link5_ID = env.robot.model.getFrameId("link5")
        self.link6_ID = env.robot.model.getFrameId("link6")

        self.create_box()

    def visualize_box(self, base_pos, base_quat):
        boxVisualShapeId = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.x_size / 2, self.y_size / 2, self.z_size / 2],
            rgbaColor=[0, 1, 0, 0.2],
        )

        self.box = p.createMultiBody(
            baseVisualShapeIndex=boxVisualShapeId,
            basePosition=[0, 0, 1],
        )

        p.resetBasePositionAndOrientation(self.box, base_pos, base_quat)

    def create_box(self):
        """
        Array : Distance from the robot to the faces of the boundary in
        [x, y, z, -x, -y, -z] axis in that order.
        np.tile --> since we have 6 links
        """

        self.A = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        )  # (6, 3)

        b = 0.5 * np.array(
            [
                [self.x_size],
                [self.y_size],
                [self.z_size],
                [self.x_size],
                [self.y_size],
                [0],
            ]
        )
        self.B = np.tile(b, 6)  # (6, 6)

    def get_link_pos(self):
        """
        Gets position of all the 6 links of the robot [x y z]
        is a 6 x 3 vector. Returns the transpose i.e. 3 x 6 vector
        """

        p_links = np.array(
            [
                self.env.robot.data.oMf[self.link1_ID].translation,
                self.env.robot.data.oMf[self.link2_ID].translation,
                self.env.robot.data.oMf[self.link3_ID].translation,
                self.env.robot.data.oMf[self.link4_ID].translation,
                self.env.robot.data.oMf[self.link5_ID].translation,
                self.env.robot.data.oMf[self.link6_ID].translation,
            ]
        ).T  # (3, 6)

        return p_links

    def check_out_of_bound(self):
        """
        Checks if the robot is within the boundary
        """
        p_links = self.get_link_pos()
        out_of_bound = np.any(self.A @ p_links >= self.B)

        return out_of_bound
