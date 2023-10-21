import numpy as np
import proxsuite


class DiffIK:
    def __init__(self):
        self.epsilon = 0.01
        self.kp_joint_centering = 1.0

        self.n = 6
        self.n_eq = 0
        self.n_ieq = 0
        self.qp = proxsuite.proxqp.dense.QP(self.n, self.n_eq, self.n_ieq)
        self.qp_initialized = False

        self.q_nominal = np.zeros((6, 1))

    def __call__(self, V_des, q, J):
        self.H, self.g = self.compute_params(V_des, q, J)

        if not self.initialized:
            self.qp.init(H=self.H, g=self.g)
            self.qp.settings.eps_abs = 1.0e-6
            self.qp.settings.max_iter = 20
            self.qp_initialized = True
        else:
            self.qp.update(H=self.H, g=self.g)

        self.qp.solve()

        return self.qp.results.x

    def compute_params(self, V_des, q, J):
        nullspace_mat = np.eye(self.n) - np.linalg.pinv(J) @ J
        nullspace_quadratic = nullspace_mat.T @ nullspace_mat

        H = 2 * (J.T @ J + nullspace_mat.T @ nullspace_mat)

        dq_nominal = self.kp_joint_centering * (self.q_nominal - q)
        g = -2 * (V_des.T @ J + dq_nominal.T @ nullspace_quadratic)

        return H, g
