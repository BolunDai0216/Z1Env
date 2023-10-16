import numpy as np

class SplineGen():

    def __init__(self, q_init, q_term, T):
        self.q_init = q_init
        self.q_term = q_term
        self.T  = T
        self.computeCoeff()

    def computeCoeff(self):
        # X = np.zeros((6,6))
        A = np.array([[0,0,0,0,0,1] ,
                      [self.T**5,self.T**4,self.T**3,self.T**2,self.T,1],
                      [0,0,0,0,1,0],
                      [5*(self.T**4),4*(self.T**3),3*(self.T**2),2*self.T,1,0],
                      [0,0,0,1,0,0],
                      [20*(self.T**3),12*(self.T**2),6*self.T,2,0,0]])    

        # q_initial = np.zeros((1,7))
        vel_ = np.zeros((2,7))
        acc_ = np.zeros((2,7))
        B= np.concatenate([self.q_init,self.q_term,vel_,acc_], axis= 0)
        
        # A.shape,  B.shape
        self.X = np.linalg.inv(A) @ B
        # print(self.X.shape)
        # return self.X
    
    def get_joint_Angles(self,t):

        # q1 = X[0,0]*t**5 + X[1,0]*t**4 + X[2,0]*t**3 + X[3,0]*t**2 + X[4,0]*t + X[5,0]
        # q2 = X[0,1]*t**5 + X[1,1]*t**4 + X[2,1]*t**3 + X[3,1]*t**2 + X[4,1]*t + X[5,1]
        # q3 = X[0,2]*t**5 + X[1,2]*t**4 + X[2,2]*t**3 + X[3,2]*t**2 + X[4,2]*t + X[5,2]
        # q4 = X[0,3]*t**5 + X[1,3]*t**4 + X[2,3]*t**3 + X[3,3]*t**2 + X[4,3]*t + X[5,3]
        # q5 = X[0,4]*t**5 + X[1,4]*t**4 + X[2,4]*t**3 + X[3,4]*t**2 + X[4,4]*t + X[5,4]
        # q6 = X[0,5]*t**5 + X[1,5]*t**4 + X[2,5]*t**3 + X[3,5]*t**2 + X[4,5]*t + X[5,5]


        # q = [q1,q2,q3,q4,q5,q6,0]
        T_coeff = np.array([[t**5 ],[t**4],[t**3],[t**2] ,[t] ,[1]]) 
        # print(T_coeff.shape)       
        q = np.transpose(self.X) @ T_coeff
        # print(q.shape)
        # q = np.concatenate(np.zeros((1,6)),q)
        return np.reshape(q,(1,7))
    
    def get_joint_Vel(self,t):

        # q1_dot = X[0,0]*5*(t**4) + X[1,0]*4*(t**3) + X[2,0]*3*(t**2)  + X[3,0]*2*t + X[4,0]*1 + X[5,0]*0
        # q2_dot = X[0,1]*5*(t**4) + X[1,1]*4*(t**3) + X[2,1]*3*(t**2)  + X[3,1]*2*t + X[4,1]*1 + X[5,1]*0
        # q3_dot = X[0,2]*5*(t**4) + X[1,2]*4*(t**3) + X[2,2]*3*(t**2)  + X[3,2]*2*t + X[4,2]*1 + X[5,2]*0
        # q4_dot = X[0,3]*5*(t**4) + X[1,3]*4*(t**3) + X[2,3]*3*(t**2)  + X[3,3]*2*t + X[4,3]*1 + X[5,3]*0
        # q5_dot = X[0,4]*5*(t**4) + X[1,4]*4*(t**3) + X[2,4]*3*(t**2)  + X[3,4]*2*t + X[4,4]*1 + X[5,4]*0
        # q6_dot = X[0,5]*5*(t**4) + X[1,5]*4*(t**3) + X[2,5]*3*(t**2)  + X[3,5]*2*t + X[4,5]*1 + X[5,5]*0
        # qdot = [q1_dot, q2_dot,q3_dot,q4_dot,q5_dot,q6_dot,0]

        T_coeff = np.array([[5*(t**4) ], [4*(t**3)] ,[3*(t**2)], [2*t] ,[1] ,[0]])
        # qdot = [q1_dot,q2_dot,q3_dot,q4_dot,q5_dot,q6_dot,0]
        qdot = np.transpose(self.X) @ T_coeff
        
        return np.reshape(qdot,(1,7))
