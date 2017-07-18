#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import argparse
import tensorflow as tf
from data_provider import get_data as YoutubeData  # noqa
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbf
import glob
import os

"""
Learning Blind Motion Deblurring
"""

SEQ_LEN = 5
BATCH_SIZE = 8
SHAPE = 128
LEVELS = 3


def ReluConv2D(name, x, out_channels, use_relu=True, kernel_shape=3, stride=1):
    if use_relu:
        x = tf.nn.relu(x, name='%s_relu' % name)
    x = Conv2D('%s_conv' % name, x, out_channels, kernel_shape=kernel_shape, stride=stride)
    return x


def ReluDeconv2D(name, x, out_channels, kernel_shape=3, stride=1):
    x = tf.nn.relu(x, name='%s_relu' % name)
    x = Deconv2D('%s_deconv' % name, x, out_channels, kernel_shape=kernel_shape, stride=stride)
    return x


def Merge(incoming_skip, ID, tensor, name):
    with tf.name_scope('Merge_%s' % name):
        if incoming_skip is None:
            # we gonna fake the skip, to allow TF to reuse variable and construct
            # for this block a senseless conv-layer
            incoming_skip_internal = tensor
        else:
            # we really want to merge both layers
            incoming_skip_internal = incoming_skip[ID]
        hs, ws = incoming_skip_internal.get_shape().as_list()[1:3]
        hl, wl = tensor.get_shape().as_list()[1:3]

        # tmp_name = resize(incoming_skip_internal, name)
        # if (hs != hl) or (ws != wl):
        #     incoming_skip_internal = tmp_name
        channels = tensor.get_shape().as_list()[3]
        tensor_internal = tf.concat([tensor, incoming_skip_internal], axis=3)
        tensor_internal = ReluConv2D(name, tensor_internal, channels, kernel_shape=1)

        if incoming_skip is None:
            # we have constructed the operation but just return the unmodified tensor itself
            # workaround for '@auto_reuse_variable_scope'
            # be aware this gives warnings "not gradient w.r.t. ..."
            return tensor
        else:
            # we return the modified tensor
            return tensor_internal


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SEQ_LEN, SHAPE, SHAPE, 3), 'blurry'),
                InputDesc(tf.float32, (None, SEQ_LEN, SHAPE, SHAPE, 3), 'sharp')]

    @auto_reuse_variable_scope
    def deblur_block(self, observation, estimate,
                     skip_temporal_in=None,
                     name=None):
        """Apply one deblur step.

        Args:
            observation: new unseen observation
            estimate: latest estimate (the image which should be improved)
            skip_temporal_in (None, optional): list of skip_connections
            skip_unet_out(None, optional): lsit of skip connections between deblurring blocks within the network .
        """

        skip_temporal_out = []  # green
        skip_unet_out = []      # grey

        with tf.name_scope("deblur_block_%s" % name):
            # be aware use_local_stat=True gives warnings
            with argscope(BatchNorm, use_local_stat=True), \
                    argscope([Conv2D, Deconv2D], nl=lambda x, name: BatchNorm(name, x)):
                inputs = tf.concat([observation, estimate], 3)

                block = ReluConv2D('d0', inputs, 32, stride=1, kernel_shape=3)

                # H x W -> H/2 x W/2
                # ---------------------------------------------------------------------
                with tf.name_scope('block_0'):
                    block = ReluConv2D('d1_0', block, 64, stride=2)
                    block_start = block
                    block = ReluConv2D('d1_1', block, 64)
                    block = ReluConv2D('d1_2', block, 64)
                    block = ReluConv2D('d1_3', block, 64, kernel_shape=1)
                    block = tf.add(block_start, block, name='block_skip_A')

                # H/2 x W/2 -> H/2 x W/2
                # ---------------------------------------------------------------------
                with tf.name_scope('block_1'):
                    block = ReluConv2D('d2_0', block, 64)
                    block_start = block
                    block = ReluConv2D('d2_1', block, 64)
                    block = ReluConv2D('d2_2', block, 64)
                    block = ReluConv2D('d2_3', block, 64, kernel_shape=1)
                    block = tf.add(block_start, block, name='block_skip_B')
                    skip_unet_out.append(block)

                # H/2 x W/2 -> H/4 x W/4
                # ---------------------------------------------------------------------
                with tf.name_scope('block_2'):
                    block = ReluConv2D('d3_0', block, 128, stride=2)
                    block_start = block
                    block = ReluConv2D('d3_1', block, 128)
                    block = ReluConv2D('d3_2', block, 128)
                    block = ReluConv2D('d3_3', block, 128, kernel_shape=1)
                    block = tf.add(block_start, block, name='block_skip_C')
                    skip_unet_out.append(block)

                # H/4 x W/4 -> H/8 x W/8
                # ---------------------------------------------------------------------
                with tf.name_scope('block_3'):
                    block = ReluConv2D('d4_0', block, 256, stride=2)
                    block_start = block
                    block = Merge(skip_temporal_in, 0, block, 'd41_s')
                    block = ReluConv2D('d4_1', block, 256)
                    block = ReluConv2D('d4_2', block, 256)
                    block = ReluConv2D('d4_3', block, 256, kernel_shape=1)
                    block = tf.add(block_start, block, name='block_skip_D')
                    skip_temporal_out.append(block)

                # H/8 x W/8 -> H/4 x W/4
                # ---------------------------------------------------------------------
                with tf.name_scope('block_4'):
                    block = ReluDeconv2D('u1_0', block, 128, stride=2, kernel_shape=4)
                    block = tf.add(block, skip_unet_out[1], name='skip01')
                    block_start = block
                    block = Merge(skip_temporal_in, 1, block, 'u1_s')
                    block = ReluConv2D('u1_1', block, 128)
                    block = ReluConv2D('u1_2', block, 128)
                    block = ReluConv2D('u1_3', block, 128)
                    block = tf.add(block, block_start, name='block_skip_E')
                    skip_temporal_out.append(block)

                # H/4 x W/4 -> H/2 x W/2
                # ---------------------------------------------------------------------
                with tf.name_scope('block_5'):
                    block = ReluDeconv2D('u2_0', block, 64, stride=2, kernel_shape=4)
                    block = tf.add(block, skip_unet_out[0], name='skip02')
                    block_start = block
                    block = Merge(skip_temporal_in, 2, block, 'u2_s')
                    block = ReluConv2D('u2_1', block, 64)
                    block = ReluConv2D('u2_2', block, 64)
                    block = ReluConv2D('u2_3', block, 64)
                    block = tf.add(block, block_start, name='block_skip_F')
                    skip_temporal_out.append(block)

                # H/2 x W/2 -> H x W
                # ---------------------------------------------------------------------
                with tf.name_scope('block_6'):
                    block = ReluDeconv2D('u3_0', block, 64, stride=2, kernel_shape=4)
                    block = ReluConv2D('u3_1', block, 64)
                    block = ReluConv2D('u3_2', block, 64)
                    block = ReluConv2D('u3_3', block, 6)
                    block = ReluConv2D('u3_4', block, 3)
                estimate = tf.add(estimate, block, name='skip03')

                return estimate, skip_temporal_out

    def _build_graph(self, input_vars):

        # some loss functions and metrics to track performance
        def l2_loss(x, y, name):
            return tf.reduce_mean(tf.squared_difference(x, y), name=name)

        def l1_loss(x, y, name):
            return tf.reduce_mean(tf.abs(x - y), name=name)

        def scaled_psnr(x, y, name):
            return symbf.psnr(128. * (x + 1.0), 128. * (y + 1.), 255, name=name)

        # centered inputs [B, T, H, W, C]
        blurry, sharp = input_vars
        blurry = blurry / 128.0 - 1
        sharp = sharp / 128.0 - 1

        l2err_list, l1err_list, psnr_list, psnr_impro_list = [], [], [], []

        estimate = blurry[:, -1, :, :, :]
        expected = sharp[:, -1, :, :, :]

        skip_temporal_out = None
        estimate_viz = []

        psnr_base = scaled_psnr(blurry[:, SEQ_LEN - 1, :, :, :], sharp[:, -1, :, :, :], name="PSNR_base")

        for t in range(1, SEQ_LEN):
            logger.info("build time step: %i" % t)

            # get observation at all scales in time step 't'
            observation = blurry[:, SEQ_LEN - t - 1, :, :, :]
            logger.info("time step: {} with input shape {}".format(t, observation[t].get_shape()))

            estimate, skip_temporal_out = \
                self.deblur_block(observation,
                                  estimate,
                                  skip_temporal_in=skip_temporal_out,
                                  name='level_%i_step_%i' % (0, t))

            l2err_list.append(l2_loss(estimate, expected, name="L2loss_t%i" % (t)))
            l1err_list.append(l1_loss(estimate, expected, name="L1loss_t%i" % (t)))
            psnr_list.append(scaled_psnr(estimate, expected, name="PSNR_t%i" % (t)))
            pi = tf.divide(psnr_list[-1], psnr_base, name="PSNR_IMPRO_t%i" % (t))
            psnr_impro_list.append(pi)

            # naming estimates for grabbing during deployment
            tf.identity((estimate + 1.0) * 128., name='estimate_t%i_l%i' % (0,t))

            estimate_viz.append(estimate)

        # just visualize original images
        with tf.name_scope('visualization'):
            estimate_viz = tf.concat(estimate_viz, axis=2)
            observed_viz = tf.concat([blurry[:, i, :, :, :] for i in range(SEQ_LEN)], axis=2)

            viz = tf.concat([observed_viz, estimate_viz, expected], axis=2, name='estimates')
            viz = 128.0 * (viz + 1.0)
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('blurry5_estimates5_expected', viz, max_outputs=max(30, BATCH_SIZE))

        # total cost is sum of all individual losses
        self.cost = tf.add_n(l2err_list, name="total_cost")
        add_moving_summary(self.cost)

        for l in range(LEVELS):
            add_moving_summary(l2err_list + l1err_list + psnr_list + psnr_impro_list)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 0.005, summary=True)
        return tf.train.AdamOptimizer(lr)


def get_config(datadir, batch_size):
    logger.auto_set_dir('n')
    lmdbs = glob.glob(os.path.join(datadir, 'train*.lmdb'))
    ds_train = [YoutubeData(lmdb, shape=(128, 128), ego_motion_size=[17, 21, 25]) for lmdb in lmdbs]
    ds_train = RandomMixData(ds_train)
    ds_train = BatchData(ds_train, BATCH_SIZE)
    ds_train = PrefetchDataZMQ(ds_train, 8)

    lmdbs = glob.glob(os.path.join(datadir, 'val*.lmdb'))
    ds_val = [YoutubeData(lmdb, shape=(128, 128), ego_motion_size=[17, 21, 25]) for lmdb in lmdbs]
    ds_val = RandomMixData(ds_val)
    ds_val = BatchData(ds_val, BATCH_SIZE)
    ds_val = FixedSizeData(ds_val, 100)
    ds_val = PrefetchDataZMQ(ds_val, 8)

    steps_per_epoch = 1000

    return TrainConfig(dataflow=ds_train,
                       callbacks=[
                           ModelSaver(),
                           InferenceRunner(ds_val, [ScalarStats('total_cost'),
                                                    ScalarStats('PSNR_IMPRO_t%i' % (SEQ_LEN - 1))])
                       ],
                       extra_callbacks=[
                           MovingAverageSummary(),
                           ProgressBar(['tower0/PSNR_base',
                                        'tower0/PSNR_IMPRO_t%i' % (SEQ_LEN - 1),
                                        'tower0/PSNR_IMPRO_t%i' % (SEQ_LEN - 1),
                                        'tower0/PSNR_IMPRO_t%i' % (SEQ_LEN - 1),
                                        ]),
                           MergeAllSummaries(),
                           RunUpdateOps()
                       ],
                       model=Model(),
                       steps_per_epoch=steps_per_epoch,
                       max_epoch=400)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--batch', help='batch-size', type=int, default=32)
    d = '/graphics/projects/scratch/wieschol/YouTubeDataset/'
    parser.add_argument('--data', help='batch-size', type=str, default=d)
    parser.add_argument('--load', help='load model')

    args = parser.parse_args()

    NR_GPU = len(args.gpu.split(','))
    with change_gpu(args.gpu):
        config = get_config(args.data, args.batch)
        if args.load:
            config.session_init = SaverRestore(args.load)
        config.nr_tower = NR_GPU
        SyncMultiGPUTrainer(config).train()
