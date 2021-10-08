import time
import control
import numpy as np


from utils import soft_shrinkage, rsolve, batched_frobenius_norm


class RkcaOrderTwoAdmm(object):
    """ADMM solver for RKCA with order 2 regularization - also known as KDRSDL
       Algorithm 1 of [1, 2].

       [1] Robust Kronecker-Decomposable Component Analysis for Low-Rank Modeling,
            M. Bahri, Y. Panagakis, and S. Zafeiriou, ICCV, 2017
       [2] Robust Kronecker Component Analysis, M. Bahri. Y. Panagakis, S. Zafeiriou,
            IEEE T-PAMI, 2019
    """
    def __init__(self,
        X,
        r=None,
        lambda_=None,
        tol=1e-7,
        maxiter=150,
        rho=1.2,
        alpha_a=1,
        alpha_b=1,
        alpha=1e-1,
        mu=None,
        mu_bar=np.inf,
        convergence_core_R=False,
    ):
        # Reference the input
        self.X = X
        # Dimensions of the input
        self.Nobs, self.n, self.m = X.shape

        self.init_if_unset(r, lambda_)

        # Parameters as passed to the constructor
        self.tol = tol
        self.maxiter = maxiter
        self.rho = rho
        self.alpha_a = alpha_a
        self.alpha_b = alpha_b
        self.alpha = alpha
        self.mu = mu
        self.mu_bar = mu_bar

        self.variables_initialized_ = False
        self.extra_variables_initialized_ = False

        if convergence_core_R:
            self.convergence_criterion = self.convergence_criterion_core_R_
        else:
            self.convergence_criterion = self.convergence_criterion_core_K_

        self.init_constants()
        self.init_mu_with_norms()


    def init_if_unset(self, r, lambda_):
        if r is None:
            self.r = np.minimum(self.n, self.m)
        else:
            self.r = r

        if lambda_ is None:
            self.lambda_ = 1 / np.sqrt(self.Nobs * np.maximum(self.n, self.m))
        else:
            self.lambda_ = lambda_


    def init_variables(self):
        if not self.variables_initialized_:
            print("Initializing variables with SVD...")
            st = time.time()

            # Sparse error
            self.E = np.zeros_like(self.X)

            # Lagrange multipliers
            self.Y = np.zeros_like(self.X)

            # Core tensor and left and right bases
            # Initialize with partial SVD
            U, S, V = np.linalg.svd(self.X, full_matrices=False)
            # SVD returns V^H but we did the math for V
            V = V.transpose(0, 2, 1)
            self.A = U[:, :, : self.r].mean(axis=0)
            self.B = V[:, :, : self.r].mean(axis=0)
            self.R = S[:, None, : self.r] * np.eye(self.r)

            self.L_R = self.A @ self.R @ self.B.T

            et = time.time()
            print(f"Done in {et-st:.3f}s")

            self.variables_initialized_ = True


    def init_constants(self):
        # Pre-compute some operations that are used often and do not change
        self.Ir = np.eye(self.r)
        self.norms_fs_X = batched_frobenius_norm(self.X)


    def init_mu_with_norms(self, rescaling_coefficient=1.25):
        self.mu = rescaling_coefficient * self.Nobs / self.norms_fs_X.sum()


    def init_extra_variables(self, rescaling_coefficient=1.25):
        if not self.extra_variables_initialized_:
            print("Initializing extra variables...")
            self.K = self.R.copy()
            self.Yk = np.zeros_like(self.R)

            self.mu_k = rescaling_coefficient * self.Nobs / batched_frobenius_norm(self.R).sum()
            print("Done")

            self.extra_variables_initialized_ = True


    def convergence_criterion_(self, delta):
        # Reconstruction error defined as the max reconstruction error
        # over the slices
        e_slices_n = batched_frobenius_norm(delta)
        e_slices_d = self.norms_fs_X
        # Max relative error per frontal slide
        e_slice = (e_slices_n / e_slices_d).max()

        # Splitting error
        e_split = (batched_frobenius_norm(self.K - self.R) / batched_frobenius_norm(self.K)).max()

        return np.maximum(e_slice, e_split)


    def convergence_criterion_core_K_(self):
        # Use K as the core variable for true sparsity, but must recompute the low-rank component
        delta = self.Xt - self.A @ self.K @ self.B.T
        return self.convergence_criterion_(delta)


    def convergence_criterion_core_R_(self):
        # Reuse cached computation from the update of the core variables
        return self.convergence_criterion_(self.delta_L_R)


    def update_A_L2(self):
        Rt = self.R.transpose(0, 2, 1)

        An = (self.S @ self.B @ Rt).sum(axis=0)
        Ad = self.mu * (self.R @ (self.B.T @ self.B) @ Rt).sum(axis=0)

        denom = self.alpha_a * self.Ir + Ad

        self.A = rsolve(An, denom)


    def update_B_L2(self):
        Bn = (self.S.transpose(0, 2, 1) @ self.A @ self.R).sum(axis=0)
        Bd = self.mu * (self.R.transpose(0, 2, 1) @ (self.A.T @ self.A) @ self.R).sum(axis=0)

        denom = self.alpha_b * self.Ir + Bd

        self.B = rsolve(Bn, denom)


    def update_E(self):
        # self.L_R should not have changed since its update in update_RY_regR_L1
        E = self.X - self.L_R + (1 / self.mu) * self.Y

        self.E = soft_shrinkage(E, self.lambda_ / self.mu)


    def update_RY_regR_L1(self):
        # B should be m x r and A n x r
        red_S = ( (self.A.T @ self.S @ self.B) + self.mu_k * self.K + self.Yk ) / self.mu_k

        U = (-self.mu / self.mu_k) * (self.A.T @ self.A)
        V = self.B.T @ self.B

        # Solve the Sylvester's equation for every slice
        R_ = np.zeros_like(self.R)
        for i in range(self.Nobs):
            R_[i] = control.dlyap(U, V, red_S[i])

        # Split variable
        K_ = soft_shrinkage(R_ - self.Yk / self.mu_k, self.alpha / self.mu_k)

        # Update the core and the split variable
        self.R = R_
        self.K = K_

        # Low-rank component and delta for use in convergence criterion
        self.L_R = self.A @ self.R @ self.B.T
        self.delta_L_R = self.Xt - self.L_R

        # Update the Lagrange multipliers
        self.Y = self.Y + self.mu * self.delta_L_R
        self.Yk = self.Yk + self.mu_k * (self.K - self.R)

        self.mu_k = np.minimum(self.mu_bar, self.rho * self.mu_k)


    def fit(self):
        self.init_variables()
        self.init_extra_variables()

        self.converged = False
        self.niter = 0
        EE = np.inf
        # Save errors for plotting
        self.err = []

        while (not self.converged) and (self.niter < self.maxiter):
            self.niter += 1

            # --------------------------------------------
            # Measure time spent in the iteration
            st = time.time()

            # Update E first
            self.update_E()

            # X tilde
            self.Xt = self.X - self.E

            # Start with A and B
            self.S = self.mu * self.Xt + self.Y
            self.update_A_L2()
            self.update_B_L2()

            # Core
            self.update_RY_regR_L1()

            # Mu (mu_k gets updated with the core because the
            # split variable is a consequence of the choice of
            #  the model order)
            self.mu = np.minimum(self.mu_bar, self.rho * self.mu)

            # Test for convergence
            EEp = EE
            EE = self.convergence_criterion()
            if EE < self.tol:
                self.converged = True

            et = time.time()
            delta_t_iter = et - st

            # --------------------------------------------

            self.err.append(EE)

            # Estimate ranks for A and B
            st = time.time()

            estim_rank_A = np.linalg.matrix_rank(self.A, tol=1e-3)
            estim_rank_B = np.linalg.matrix_rank(self.B, tol=1e-3)

            et = time.time()
            delta_t_rank = et - st

            print(f"[{self.niter:03d}] mu = {self.mu:.3e} max err. = {EE:.3e} | rk(A, 1e-3) = {estim_rank_A} rk(B, 1e-3) = {estim_rank_B} (est. in {delta_t_rank:.3e}s) | Iter time: {delta_t_iter:.3f}s")

        return self.converged, self.niter, self.err


    def get_reconst_R(self):
        self.L_R = self.A @ self.R @ self.B.T

        return self.L_R, self.E


    def get_reconst_K(self):
        self.L_K = self.A @ self.K @ self.B.T

        return self.L_K, self.E


    def get_reconst(self):
        return self.get_reconst_K()