import argparse
import cv2
import numpy as np
import glob as glob

"""
re-implementation:
Removing Camera Shake via Weighted Fourier Burst Accumulation
"""


def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def fba(stack, p=17):
        m, n, = stack[0].shape
        c = 1
        hu_p = np.zeros((m, n)) + 0j
        w = np.zeros((m, n))

        for img in stack:
                hv_i = np.zeros((m, n, c)) + 0j
                hv_i = np.fft.fft2(np.array(img).astype(float), axes=[0, 1])
                w_i = abs(hv_i)
                hu_p = hu_p + (w_i**p + 0j) * hv_i
                w = w + w_i**p

        u_p = np.zeros((m, n, c)) + 0j
        u_p = np.fft.ifft2(hu_p[:, :] / (w), axes=[0, 1])
        return np.clip(u_p.real, 0, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', help='images used for fba (in glob)', required=True)
    parser.add_argument('--out', help='output of fba', default='fba_result.png')
    parser.add_argument('--p', action='parameter of fba', type=int, default=17)
    args = parser.parse_args()

    images = glob.glob(args.pattern)
    batch = [] * len(images)
    for i in range(len(images)):
        batch[i] = rgb2gray(cv2.imread(images[i]) / 255.)

    sharp = fba(batch, args.p)
    cv2.imwrite(args.out, sharp.astype(float))
