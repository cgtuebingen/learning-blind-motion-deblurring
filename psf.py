#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

"""
Sampling PSF kernels for synthetic camera shake.

see "A machine learning approach for non-blind image deconvolution" (CVPR2013)
for details.
"""

import numpy as np
import cv2


class PSF(object):
    """Generator for synthetic PSF kernels.

    Usage:
        p = PSF(kernel_size=17)
        for psf in p.sample():
            print(psf)
    """
    @staticmethod
    def cov_materniso(x, z, hyp=np.log([0.3, 1 / 4.]), d=3, i=None):
        ell = np.exp(hyp[0])
        sf2 = np.exp(2. * hyp[1])

        def f(t):
            return 1. + t

        def df(t):
            return t

        def m(t, f):
            return f(t) * np.exp(-t)

        def dm(t, f):
            return df(t) * np.exp(-t) * t

        def sq_dist(a, b):
            if len(a.shape) == 1:
                a = np.reshape(a, (-1, 1))
            if len(b.shape) == 1:
                b = np.reshape(b, (-1, 1))
            return (a * a) + (b * b).T - 2 * np.matmul(a, b.T)
            # return np.sum(a*a, 0).T[:, None] + np.sum(b*b, 0) - 2 * np.matmul(a.T, b)

        a = (np.sqrt(d) / ell) * x
        b = (np.sqrt(d) / ell) * z
        k = sq_dist(a, b)
        k = sf2 * m(np.sqrt(k), f)
        return k

    def __init__(self, kernel_size=7, trajectory_dim=2, num_interpolation=1000):
        """Init a new PSF generator

        Args:
            kernel_size (int, optional): size of psf kernel
            trajectory_dim (int, optional): dimension of trajectory space
            num_interpolation (int, optional): number of samples on trajectory
        """
        super(PSF, self).__init__()
        self.kernel_size = kernel_size
        self.trajectory_dim = trajectory_dim
        self.num_interpolation = num_interpolation

        self.means = .5 * np.ones((self.num_interpolation, 1))

        t = np.linspace(0, 1, num_interpolation)
        covs = PSF.cov_materniso(t, t)
        self.C = np.linalg.cholesky(covs).T

    def sample(self):
        """Yield a generated PSF
        """
        max_length = self.kernel_size
        sf = self.kernel_size

        def centerofmass(f):
            sf = f.shape[0]
            f = np.abs(f)
            f = f / float(f.sum())
            i = np.sum(np.matmul(np.arange(1, sf + 1), f))
            j = np.sum(np.matmul(np.arange(1, sf + 1), f.T))
            return np.array([i, j])

        def circshift(a, shift):
            a = np.roll(a, shift[0] % a.shape[0], axis=0)
            a = np.roll(a, shift[1] % a.shape[1], axis=1)
            return a

        sample = np.zeros((self.num_interpolation, self.trajectory_dim))

        while True:
            for i in range(self.trajectory_dim):
                sample[:, i] = (np.matmul(self.C.T, np.random.randn(self.C.shape[0], 1)) + self.means).flatten()

            max_length = sf
            scaled = np.round((max_length - 1) * sample + 1 + 0.1)

            if ((scaled < 1).any() or (scaled > max_length).any()):
                # out of bounds
                continue

            scaled = scaled.astype(int)

            # Converts trajectory to blur kernel.
            f = np.zeros((max_length, max_length))
            for i in scaled:
                f[i[0] - 1, i[1] - 1] += 1

            # Centers kernel up to one pixel precision, could also cause out of bounds.
            shift = np.array([max_length + 1, max_length + 1]) / 2.0
            shift -= np.round(centerofmass(f) + 0.1)
            shift = shift.astype(int)
            f = circshift(f, shift)

            # Test if shift actually caused out of bounds.
            test_center = np.round(centerofmass(f) + 0.1)
            if np.sum(test_center - [(max_length + 1) // 2, (max_length + 1) // 2]) == 0:
                # smooth kernel slightly
                # fhh = centerofmass(f)
                # print fhh
                f = cv2.GaussianBlur(f, (3, 3), sigmaX=0.3, sigmaY=0.3)
                f /= f.sum()
                # subpixel shift
                for k in range(3):
                    ff = np.copy(f)
                    shift = np.array([max_length + 1., max_length + 1.]) / 2.0 - centerofmass(ff)
                    affine_warp = np.array([[1., 0., shift[1]],
                                            [0., 1., shift[0]]]).astype("float32")

                    ff = cv2.warpAffine(ff.astype("float32"), affine_warp, (max_length, max_length))
                    # print k, centerofmass(ff), ff.sum(), (self.kernel_size + 1) / 2.
                if abs(ff.sum() - 1) > 1e-8:
                    continue

                ff /= ff.max()
                ff /= ff.sum()
                yield ff
            else:
                continue


if __name__ == '__main__':
    # usage is like this
    p = PSF(kernel_size=35)
    for k in p.sample():
        # for visualization --> rescale
        viz = np.copy(k)
        viz -= viz.min()
        viz /= viz.max()
        viz = cv2.resize(viz, (0, 0), fx=4, fy=4)
        cv2.imshow('PSF', viz)
        cv2.waitKey(0)
