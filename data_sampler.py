#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import numpy as np

import tensorpack as tp
from tensorpack import *
import cv2
import video
import argparse
import os
from glob import glob

"""
Sampling a burst of consecutive frames for motion blur.

Example:

    python data_sampler.py --pattern '/graphics/scratch/wieschol/YouTubeDataset/train/*_blurry.mp4' \
        --lmdb /tmp/train.lmdb
"""


def get_video(pattern, passes=10, rng=None):

    # they cause a 'moov atom not found' warning
    # ignores = ['PsrPTpg6mNo', 'SAyOr2hTRkM', 'wxZO2UjBw']

    video_list = glob(pattern)
    if rng:
        rng.shuffle(video_list)

    for _ in range(passes):
        for fn in video_list:

            # # ignore some videos
            # for i in ignores:
            #     if i in fn:
            #         continue

            sharp_video = fn.replace("_blurry.mp4", "_sharp.mp4")
            if os.path.isfile(sharp_video):
                yield fn


def get_random_sharp_frames(fn_seq, window_size=5, avg_frames=1, dark_tresh=8,
                            number_of_picked_frames=100, max_attemps=100, rng=None):
    """Find good frames from videos (poor mans)

    Args:
        fn (str): path to video (should end with __blurry.mp4)
        window_size (int, optional): number of subsequent frame
        avg_frames (int, optional): number subframes to average (param "L" in the paper)
        dark_tresh (int, optional): some magic treshold (first guess)
        number_of_picked_frames (int, optional): number of frames we wafrom this video
        max_attemps (int, optional): a hyperparameter describing our patience

    Yields:
        TYPE: tuple (blurry, sharp)
    """

    try:
        for fn in fn_seq:
            assert avg_frames % 2 == 1

            if not os.path.isfile(fn):
                raise video.ReadFrameException("video %s does not exists" % fn)

            fn_blurry = fn.replace("_blurry.mp4", "_sharp.mp4")
            if not os.path.isfile(fn_blurry):
                raise video.ReadFrameException("video %s does not exists" % fn_blurry)

            print fn

            blurry_video = video.Reader(fn)
            sharp_video = video.Reader(fn_blurry)

            frameIdxs = (blurry_video.frames - window_size * avg_frames * 2) // (2 * avg_frames)
            frameIdxs = np.arange(frameIdxs) * (2 * avg_frames) + window_size * avg_frames
            if rng:
                rng.shuffle(frameIdxs)

            used_frames = 0
            attemps = 0

            # guess a frame by random
            for frameId in frameIdxs:

                # get to that location
                blurry_video.jump(frameId)
                sharp_video.jump(frameId)

                # start collection from current position
                sharp_frames = []
                blurry_frames = []

                # assume it is a nice one
                valid_frame = True
                attemps += 1

                if number_of_picked_frames < 0:
                    break

                if attemps > max_attemps:
                    yield None

                # now collect subsequent frames
                for _ in xrange(window_size):
                    reference_sharp = None
                    subframes = []

                    current_sharp = None
                    last_sharp = None

                    # subframes averaging
                    for c, a in enumerate(xrange(-avg_frames + 1, avg_frames)):

                        # do we already have a sharp one
                        if current_sharp:
                            last_sharp = current_sharp

                        current_sharp = sharp_video.read()[:, :, [2, 1, 0]]
                        curent_blurry = blurry_video.read()[:, :, [2, 1, 0]]

                        if curent_blurry.shape[2] is not 3:
                            raise ReadFrameException('blurry frame has no 3 channels {}'.format(curent_blurry.shape))
                        if current_sharp.shape[2] is not 3:
                            raise ReadFrameException('blurry frame has no 3 channels {}'.format(current_sharp.shape))
                        if curent_blurry.shape[0] < 100 or curent_blurry.shape[1] < 100:
                            raise ReadFrameException('blurry frame is to small {}'.format(curent_blurry.shape))
                        if current_sharp.shape[0] < 100 or current_sharp.shape[1] < 100:
                            raise ReadFrameException('sharp frame is to small {}'.format(curent_blurry.shape))

                        # test if too dark
                        if current_sharp.astype("float32").mean() < dark_tresh:
                            print "too dark", current_sharp.mean(), "vs", dark_tresh
                            valid_frame = False
                            break

                        # test too similar
                        if a > 0:
                            if np.mean((last_sharp.astype("float32") - current_sharp.astype("float32"))**2) < 0.2:
                                print "too similar"
                                valid_frame = False
                                break

                        # middle of average
                        if a == 0:
                            reference_sharp = current_sharp.astype(float)
                        subframes.append(curent_blurry.astype(float))

                    if not valid_frame:
                        break

                    used_frames += 1
                    if valid_frame:
                        blurry_frames.append(np.array(subframes).mean(axis=0))
                        sharp_frames.append(reference_sharp)

                if len(blurry_frames) == window_size:
                    number_of_picked_frames -= 1
                    yield [np.array(blurry_frames).astype('uint8'), np.array(sharp_frames).astype('uint8')]
    except video.ReadFrameException as e:
        print(e)
    except Exception as e:
        print e


def get_good_patches(frame_gen, patch_size=512, tresh=0.4, number_of_picked_patches=10, rng=None):
    for frames in frame_gen:
        for _ in range(number_of_picked_patches):
            blurry, sharp = frames
            crop_shape = (patch_size, patch_size)
            orig_shape = blurry.shape[1:3]

            # randomly sample some patches
            diffh = orig_shape[0] - crop_shape[0]
            diffw = orig_shape[1] - crop_shape[1]
            if diffh < 1:
                print "diffh is too small {} in shapes {} / {}".format(diffh, orig_shape, crop_shape)
                break
            if diffw < 1:
                print "diffw is too small {} in shapes {} / {}".format(diffw, orig_shape, crop_shape)
                break

            r_h = np.random.randint(diffh) if not rng else rng.randint(diffh)
            h0 = 0 if diffh == 0 else r_h

            r_w = np.random.randint(diffw) if not rng else rng.randint(diffw)
            w0 = 0 if diffw == 0 else r_w

            blurry_patches = blurry[:, h0:h0 + crop_shape[0], w0:w0 + crop_shape[1], :]
            sharp_patches = sharp[:, h0:h0 + crop_shape[0], w0:w0 + crop_shape[1], :]
            # now some tests

            # # patch is too similar
            # mse = np.mean((blurry_patches - sharp_patches)**2)
            # if mse < 1:
            #     pass
            # print "mse", mse

            dark_mean = np.mean((blurry_patches))
            if dark_mean < 20:
                print "--> too dark {} vs. {}".format(dark_mean, 20)
                continue
            # print "dark_mean", dark_mean

            # reject gradients (content test)
            # --------------------------------------------------
            def image_gradient(x):
                if len(x.shape) == 2:
                    gx, gy = np.gradient(x)
                    return gx, gy
                else:
                    gx, gy, gz = np.gradient(x)
                    return gy, gz

            # print (sharp_patches[0])  # It reads the first patch from a sequence of 5 patches? Matrix was output

            dx, dy = image_gradient(sharp_patches[0])

            dx = np.sum((np.sign(np.abs(dx) - 0.05) + 1.) / 2.)
            dy = np.sum((np.sign(np.abs(dy) - 0.05) + 1.) / 2.)
            ps = sharp.shape[0] * sharp.shape[1]

            if (dx < tresh * ps) or (dy < tresh * ps):
                print "--> grad dx {} vs. {}".format(dx, tresh * ps)
                print "--> grad dy {} vs. {}".format(dy, tresh * ps)
                continue
            else:
                pass
                # print "dx, dy, ps", dx, dy, ps

            # reject psnr input if not blurry enought
            # --------------------------------------------------
            def psnr(prediction, ground_truth, maxp=None):
                def log10(x):
                    numerator = np.log(x)
                    denominator = np.log(10.)
                    return numerator / denominator

                mse = np.mean((prediction - ground_truth) ** 2)
                try:
                    psnr = -10 * log10(mse)
                    if maxp:
                        psnr += 20.0 * log10(maxp)
                    return psnr
                except Exception:
                    return 100000000

            current_psnr = psnr(blurry_patches[0, ...], sharp_patches[0, ...], 255.)
            if current_psnr < 40.:
                yield [blurry_patches, sharp_patches]
            else:
                print "PSNR to hight {} vs. {}".format(current_psnr, 40.)


class VideoPatchesFlow(tp.dataflow.RNGDataFlow):
    """Create a burst of"""
    def __init__(self, pattern, window_size=5, nr_examples=10):
        super(VideoPatchesFlow, self).__init__()

        self.pattern = pattern
        self.window_size = window_size
        self.nr_examples = nr_examples
        from tensorpack.utils import get_rng
        self.rng = get_rng(self)

    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)

    def get_data(self):
        def encoder(img):
            return np.asarray(bytearray(cv2.imencode('.jpg', img)[1].tostring()), dtype=np.uint8)

        video_list = get_video(self.pattern, passes=10000, rng=self.rng)
        frame_list = get_random_sharp_frames(video_list, window_size=self.window_size,
                                             number_of_picked_frames=30, rng=self.rng)
        for b, s in get_good_patches(frame_list, number_of_picked_patches=10, rng=self.rng):
            values = []
            for i in range(self.window_size):
                b_enc = encoder(b[i])
                values.append(b_enc)

            for i in range(self.window_size):
                s_enc = encoder(s[i])
                values.append(s_enc)
            yield values
            self.nr_examples -= 1
            if self.nr_examples == 0:
                break

    def size(self):
        return self.nr_examples


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb', type=str, help='path to lmdb')
    parser.add_argument('--action', type=str, help='path to lmdb', default='create')
    # '/graphics/scratch/wieschol/YouTubeDataset/train/*_blurry.mp4'
    parser.add_argument('--pattern', type=str, help='pattern for blurry videos', required='True')
    parser.add_argument('--num', type=int, help='number of bursts', required='True')
    args = parser.parse_args()

    df = VideoPatchesFlow(args.pattern, nr_examples=args.num)

    if args.action == 'create':
        assert args.lmdb is not None
        df = PrefetchDataZMQ(df, nr_proc=32)
        dftools.dump_dataflow_to_lmdb(df, args.lmdb)

    if args.action == 'debug':
        class Decoder(MapData):
            """compress images into JPEG format"""
            def __init__(self, df):
                def func(dp):
                    return [cv2.imdecode(np.asarray(bytearray(i), dtype=np.uint8), cv2.IMREAD_COLOR) for i in dp]
                super(Decoder, self).__init__(df, func)
        df = Decoder(df)
        df.reset_state()

        for dp in df.get_data():
            nr = len(dp)
            blurry = dp[:nr // 2]
            sharp = dp[nr // 2:]

            stacked_blurry = np.hstack(blurry)
            stacked_sharp = np.hstack(sharp)

            cv2.imshow('blurry', stacked_blurry)
            cv2.imshow('sharp', stacked_sharp)
            cv2.waitKey(0)
