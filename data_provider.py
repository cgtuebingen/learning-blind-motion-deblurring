#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

from tensorpack import *
import tensorpack as tp
import argparse
import cv2
import numpy as np
import psf
from scipy import ndimage

"""
Usage:

python data_provider.py --lmdb mydb2.lmdb
"""


class PSF(tp.dataflow.DataFlow):
    """TensorPack dataflow proxy for PSF-sampler

    Attributes:
        kernel_shape (int): size of PSF kernel
        multiple (int): number of psf for one step (could be re-written by tp.BatchData)
        psf_gen (python-generator): generator producing PSF samples
    """
    def __init__(self, kernel_shape=7, multiple=5):
        self.kernel_shape = kernel_shape
        self.multiple = multiple
        self.psf_gen = psf.PSF(kernel_size=kernel_shape)

    def reset_state(self):
        pass

    def size(self):
        return 100000000

    def get_data(self):
        sampler = self.psf_gen.sample()
        while True:
            k = []
            for _ in range(self.multiple):
                k.append(next(sampler))
            yield k


class Blur(tp.dataflow.DataFlow):
    """Apply blur from SPF kernels to incoming images.

    This yields [blurry1, blurry2, ... blurry5, sharp1, sharp2, ..., sharp5]

    Attributes:
        ds_images: dataflow producing image-bursts (should already contain motion blur).
        ds_psf: dataflow producing psf kernels
    """
    def __init__(self, ds_images, ds_psf):
        self.ds_images = ds_images
        self.ds_psf = ds_psf

    def reset_state(self):
        self.ds_images.reset_state()
        self.ds_psf.reset_state()

    def size(self):
        return self.ds_images.size()

    def get_data(self):

        image_iter = self.ds_images.get_data()
        psf_iter = self.ds_psf.get_data()

        for dp_image in image_iter:

            # sample camera shake kernel
            dp_psf = next(psf_iter)

            # synthesize ego-motion
            for t, k in enumerate(dp_psf):
                blurry = dp_image[t]
                for c in range(3):
                    blurry[:, :, c] = ndimage.convolve(blurry[:, :, c], k, mode='constant', cval=0.0)
                dp_image[t] = blurry

            yield dp_image


def get_lmdb_data(lmdb_file):

    class Decoder(MapData):
        """compress images into JPEG format"""
        def __init__(self, df):
            def func(dp):
                return [cv2.imdecode(np.asarray(bytearray(i), dtype=np.uint8), cv2.IMREAD_COLOR) for i in dp]
            super(Decoder, self).__init__(df, func)

    ds = LMDBDataPoint(lmdb_file, shuffle=True)
    ds = Decoder(ds)
    return ds


def get_data(lmdb_file, shape=(256, 256), ego_motion_size=[17, 25, 35, 71]):

    # s = (shape[0] + 2 * max(ego_motion_size), shape[1] + 2 * max(ego_motion_size))
    s = (306, 306)

    ds_img = get_lmdb_data(lmdb_file)
    # to remove hints from border-handling we crop a slightly larger regions ...
    ds_img = AugmentImageComponents(ds_img, [imgaug.RandomCrop(s)], index=range(10), copy=True)
    # .. and then apply the PSF kernel ....

    ds_psf = [PSF(kernel_shape=m) for m in ego_motion_size]
    ds_psf = RandomChooseData(ds_psf)

    ds = Blur(ds_img, ds_psf)
    # ... before the final crop
    ds = AugmentImageComponents(ds, [imgaug.CenterCrop(shape)], index=range(10), copy=True)

    def combine(x):
        nr = len(x)
        blurry = np.array(x[:nr // 2])
        sharp = np.array(x[nr // 2:])
        return [blurry, sharp]
    ds = MapData(ds, combine)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, help='path to lmdb', required='True')
    parser.add_argument('--num', type=int, help='display window', default=5)
    parser.add_argument('--show', help='display window instead of writing', action='store_true')

    args = parser.parse_args()

    ds = get_data(args.lmdb, shape=(256, 256), ego_motion_size=[17, 25, 35, 71])
    ds.reset_state()

    for counter, dp in enumerate(ds.get_data()):
        # from IPython import embed
        # embed()

        blurry = dp[0]
        blurry = [blurry[i, ...] for i in range(5)]
        blurry = np.concatenate(blurry, axis=1)

        sharp = dp[1]
        sharp = [sharp[i, ...] for i in range(5)]
        sharp = np.concatenate(sharp, axis=1)

        out = np.concatenate([blurry, sharp], axis=1)[:, :, ::-1]

        if args.show:
            cv2.imshow('stacked_blurry', out)
            cv2.waitKey(0)
        else:
            cv2.imwrite('/tmp/stacked_data_%i.jpg' % counter, out)

        if counter > args.num:
            break
