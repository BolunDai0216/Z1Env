import numpy as np
import pybullet as p
import pinocchio as pin
import time
# from Z1Env import Z1Sim

class BoundaryDetection:
    def __init__(self,boundary_dimensions,env):

        '''
        Saves the dimension as length , width and height of the box
        Fetches the Frame ID of each link mentioned in the robots/z1.urdf
        '''

        # env = Z1Sim(render_mode="human")
        self.env = env
        self.length = boundary_dimensions[0]
        self.width = boundary_dimensions[1]
        self.height = boundary_dimensions[2]
    
        self.link1_ID =env.robot.model.getFrameId("link1")
        self.link2_ID = env.robot.model.getFrameId("link2")
        self.link3_ID =env.robot.model.getFrameId("link3")
        self.link4_ID = env.robot.model.getFrameId("link4")
        self.link5_ID = env.robot.model.getFrameId("link5")
        self.link6_ID = env.robot.model.getFrameId("link6")

        # self.norm_vec = np.array([[1,0,0],
        #                     [0,1,0],
        #                     [0,0,1],
        #                     [-1,0,0],
        #                     [0,-1,0],
        #                     [0,0,-1]])

    def build_boundary(self):
        boundary_position = [0, 0, 0]

        # Create the boundary collision shape
        boundary_id = p.createVisualShape(p.GEOM_BOX, halfExtents= [self.length/2 , self.width/2 , self.height/2] , rgbaColor = [1,0,0,0.2])

        # Create the boundary object
        boundary = p.createMultiBody(baseVisualShapeIndex=boundary_id, basePosition=boundary_position) 
        p.changeDynamics(boundary, -1, mass=0)  # Make the boundary static

        # Configure collision filtering to prevent self-collisions
        robot_arm_collision_filter_group = 1
        boundary_collision_filter_group = 2

        p.setCollisionFilterGroupMask(boundary, -1, boundary_collision_filter_group, boundary_collision_filter_group)
        p.setCollisionFilterPair(boundary, -1, boundary_collision_filter_group, robot_arm_collision_filter_group, 0)

    def distance_from_robot_to_boundary(self):

        '''
        Array : Distance from the robot to the faces of the boundary in [x,y,z,-x ,-y,-z] axis in that order
        np.tile --> since we have 6 links
        '''

        B = np.array([[self.length/2],
                [self.width/2],
                [self.height/2],
                [self.length/2],
                [self.width/2],
                [0]])
        B = np.tile(B,6)

        return B

    def get_linkPose(self):
        '''
        Gets position of all the 6 links of the robot [x y z ] is a 6x3 vector
        Returns the transpose i.e. 3x6 vector
        '''

        
        p_links = np.array([
                    self.env.robot.data.oMf[self.link1_ID].translation,
                    self.env.robot.data.oMf[self.link2_ID].translation,
                    self.env.robot.data.oMf[self.link3_ID].translation,
                    self.env.robot.data.oMf[self.link4_ID].translation,
                    self.env.robot.data.oMf[self.link5_ID].translation,
                    self.env.robot.data.oMf[self.link6_ID].translation])

        p_links = np.transpose(p_links)

        return p_links







    