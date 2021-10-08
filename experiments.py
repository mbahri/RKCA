import scipy.io
import numpy as np

import matplotlib.pyplot as plt

import skimage.io as io
import skimage.util as U
import skimage.data as D
import skimage.exposure as E

import os


def image_file_sp(path, noise_level=0.1):
    X = io.imread(path)
    X = E.rescale_intensity(X, out_range=(0, 1))

    X_noisy = U.random_noise(X, mode="s&p", amount=noise_level)

    return X, X_noisy


def facade_sp(noise_level=0.1):
    path = os.path.join("data", "facade.png")
    return image_file_sp(path, noise_level=noise_level)


def yale_sp(noise_level=0.1):
    data_path = os.path.join("data", "yaleb10_full_res.mat")
    X = scipy.io.loadmat(data_path)["X"].transpose(2, 0, 1)

    X_noisy = U.random_noise(X, mode="s&p", amount=noise_level)

    return X, X_noisy


def hall_bg():
    data_path = os.path.join("data", "hall.mat")
    data_raw = scipy.io.loadmat(data_path)

    O = data_raw["vid2"].transpose(2, 0, 1)
    GT = data_raw["GT"].transpose(2, 0, 1)
    GT_frames = data_raw["GT_frames"] - data_raw["frames"][1] + 1


def visualize_slices(GT, noisy, low_rank, sparse):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(GT)
    plt.subplot(1, 4, 2)
    plt.imshow(noisy)
    plt.subplot(1, 4, 3)
    plt.imshow(low_rank)
    plt.subplot(1, 4, 4)
    plt.imshow(sparse)

    plt.show()


def plot_convergence(errors):
    x = np.arange(len(errors))
    y = np.array(errors)

    plt.figure()
    plt.semilogy(x, y)
    plt.title("Convergence criterion")
    plt.show()


def post_process_color(X, L_, E_):
    L_ = E.rescale_intensity(L_.transpose(1, 2, 0), (0, 1))
    E_ = E.rescale_intensity(E_.transpose(1, 2, 0), (0, 1))

    return X.transpose(1, 2, 0), L_, E_


def pre_process_color(X):
    return X.transpose(2, 0, 1)


def post_process_no_op(X, L, E):
    return X, L, E


def pre_process_no_op(X):
    return X
