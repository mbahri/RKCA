import numpy as np


def soft_shrinkage(E, lambda_):
    return np.maximum(np.abs(E) - lambda_, 0) * np.sign(E)


def rsolve(B, A):
    sol_t = np.linalg.solve(A.T, B.T)
    return sol_t.T


def batched_frobenius_norm(X):
    "Return the Frobenius norm of every frontal slice of the 3D tensor X as an array"
    return np.linalg.norm(X.reshape(X.shape[0], -1), axis=1)
