import numpy as np


class SplineGen:
    def __init__(self, q_init, q_term, T):
        """
        Generates a spline from q_init to q_term in time T.

        Args:
            q_init (numpy.array): Initial joint angle.
            q_term (numpy.array): Final joint angle.
            T (float): Time to complete the spline.
        """
        self.q_init = q_init
        self.q_term = q_term
        self.T = T
        self.computeCoeff()

    def computeCoeff(self):
        """
        Computes the coefficients of the spline.

        Let the spline be defined as:
        q(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
        dq(t) = 5*a5*t^4 + 4*a4*t^3 + 3*a3*t^2 + 2*a2*t + a1
        ddq(t) = 20*a5*t^3 + 12*a4*t^2 + 6*a3*t + 2*a2
        """
        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [self.T**5, self.T**4, self.T**3, self.T**2, self.T, 1],
                [0, 0, 0, 0, 1, 0],
                [
                    5 * (self.T**4),
                    4 * (self.T**3),
                    3 * (self.T**2),
                    2 * self.T,
                    1,
                    0,
                ],
                [0, 0, 0, 1, 0, 0],
                [20 * (self.T**3), 12 * (self.T**2), 6 * self.T, 2, 0, 0],
            ]
        )

        dq_init, dq_term = np.zeros((1, 7)), np.zeros((1, 7))
        ddq_init, ddq_term = np.zeros((1, 7)), np.zeros((1, 7))

        B = np.concatenate(
            [self.q_init, self.q_term, dq_init, dq_term, ddq_init, ddq_term], axis=0
        )

        self.X = np.linalg.inv(A) @ B

    def get_q(self, t):
        """
        Returns the desired joint angle at time t.

        Args:
            t (float): Time in seconds.

        Returns:
            numpy.array: Desired joint angle at time t.
        """
        T_coeff = np.array([[t**5], [t**4], [t**3], [t**2], [t], [1]])
        q = np.transpose(self.X) @ T_coeff

        return q

    def get_dq(self, t):
        """
        Returns the desired joint velocity at time t.

        Args:
            t (float): Time in seconds.

        Returns:
            numpy.array: Desired joint velocity at time t.
        """
        T_coeff = np.array(
            [[5 * (t**4)], [4 * (t**3)], [3 * (t**2)], [2 * t], [1], [0]]
        )
        dq = np.transpose(self.X) @ T_coeff

        return dq
