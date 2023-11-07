import pybullet as p
import pybullet_data
import time

import numpy as np
import pinocchio as pin

from Z1Env import BoundaryDetection,Z1Sim, getDataPath

def main():
    env = Z1Sim(render_mode="human")
    info = env.reset()

    package_directory = getDataPath()
    urdf_search_path = package_directory + "/robots"
    p.setAdditionalSearchPath(urdf_search_path)

    dimensions  = [1,1,1]
    wall = BoundaryDetection(dimensions, env)

    wall.build_boundary()

    norm_vec = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,1],
                        [-1,0,0],
                        [0,-1,0],
                        [0,0,-1]])

    for i in range(10000):
        q_des = np.array([0.0, 0.625, -0.424, 0.1, 0.1, 0.1, -0.1])
        dq_des = np.zeros(7)
        tau = 5 * (q_des - info["q"]) + 0.5 * (dq_des - info["dq"]) + info["G"]
    
        Arm_to_boundary = wall.distance_from_robot_to_boundary()

        #Checking amount of time it takes to fetch the pos of the robot
        start = time.time()

        
        p_links = wall.get_linkPose()

        print(f'Time :{ time.time() - start}')

        # 6xn @ n x6 = 6x6----> n= no. of links = 6
        if np.all(norm_vec @ p_links <= Arm_to_boundary) == True:
            print("Safe")
        else:
            print("Not safe")
            
        # Send joint commands to motor
        info = env.step(tau)
        time.sleep(1e-3)

    env.close()

if __name__ == "__main__":
    main()
