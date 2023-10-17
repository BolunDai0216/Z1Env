import time

import numpy as np

from Z1Env.z1_env import Z1Sim

from spline_gen import SplineGen

def main(): 
    env = Z1Sim(render_mode="human")
    info = env.reset()

    step = 0.01
    qf =np.array([[0,0.671,-0.500,-0.144,0,0.235,0]])

    Kp = np.array([[20, 20, 20, 2, 2, 2,2]])
    Kd = np.array([[0.02, 0.02, 0.02, 0.2, 0.2, 0.2,0.2]])
    T= 10

    spline = SplineGen(info["q"].reshape((1,7)),qf,T)
  
    for j in range(20000):
        t = step * j

        if t >= 10.0:
            t = 10.0

        q = spline.get_joint_Angles(t)
        
        qdot = spline.get_joint_Vel(t)
        
        tau = Kp * (q - info["q"]) +  Kd * (qdot - info["dq"]) + info["G"]
        # print(tau.shape)
        # tau = np.reshape(tau, (7,1))

        info = env.step(np.transpose(tau)) 

        time.sleep(1e-3)
        



if __name__ == "__main__":
    main()     