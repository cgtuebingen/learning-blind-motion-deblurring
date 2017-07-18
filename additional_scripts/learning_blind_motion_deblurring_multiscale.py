#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Meenal Baheti, Patrick Wieschollek

import argparse
import tensorflow as tf
from data_provider import get_data as YoutubeData  # noqa
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbf
import glob

"""
Learning Blind Motion Deblurring (Multi -Scale version)
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


def resize(small, name):
    """resize "small" to shape of "large"
    """
    # # just resize to reduce parameters from 7043997 to ... TODOOOOOOOOO
    # return resize_by_factor(x, 2)
    with tf.variable_scope('resize'):
        out_channels = small.get_shape().as_list()[3]
        # need to add name as argument in function.
        small = Deconv2D('spatial_skip_deconv_%s' % name, small, out_channels, kernel_shape=4, stride=2)
        small = tf.nn.relu(small, name='spatial_skip_relu_%s' % name)
        return small


def resize_by_factor(x, f):
    with tf.name_scope('resize'):
        """resize "small" to shape of "large"
        """
        height, width = x.get_shape().as_list()[1:3]
        return tf.image.resize_images(x, [int(height * f), int(width * f)])


def Merge(incoming_skip, ID, tensor, name):
    with tf.name_scope('Merge_%s' % name):
        if incoming_skip is None:
            # we gonna fake the skip, to allow TF reuse variable and construct
            # for this block a senseless conv
            incoming_skip_internal = tensor
        else:
            # we really want to merge both layers
            incoming_skip_internal = incoming_skip[ID]
        hs, ws = incoming_skip_internal.get_shape().as_list()[1:3]
        hl, wl = tensor.get_shape().as_list()[1:3]

        tmp_name = resize(incoming_skip_internal, name)
        if (hs != hl) or (ws != wl):
            incoming_skip_internal = tmp_name
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
                     skip_spatial_in=None,
                     name=None):
        """Apply one deblur step.

        Args:
            observation: new unseen observation
            estimate: latest estimate
            skip_temporal_in (None, optional): list of skip_connections
            skip_spatial_in(None, optional): list of connections between multi-scaled layers
            skip_unet_out(None, optional): lsit of skip connections between deblurring blocks within the network .
        """

        skip_temporal_out = []  # green
        skip_spatial_out = []   # from resized
        skip_unet_out = []      # grey

        with tf.name_scope("deblur_block_%s" % name):
            # be aware use_local_stat=True gives warnings
            with argscope(BatchNorm, use_local_stat=True), \
                    argscope([Conv2D, Deconv2D], nl=lambda x, name: BatchNorm(name, x)):
                inputs = tf.concat([observation, estimate], 3)

                block = ReluConv2D('d0', inputs, 32, stride=1, kernel_shape=3)

                # H x W -> H/2 x W/2
                # ---------------------------------------------------------------------
                block = ReluConv2D('d1_0', block, 64, stride=2)
                block = Merge(skip_temporal_in, 2, block, 'd11_s')
                block_start = block
                block = ReluConv2D('d1_1', block, 64)
                block = ReluConv2D('d1_2', block, 64)
                block = ReluConv2D('d1_3', block, 64, kernel_shape=1)
                block = tf.add(block_start, block, name='block_skip_A')
                skip_spatial_out.append(block)
                block = Merge(skip_spatial_in, 0, block, 'd14_s')

                # H/2 x W/2 -> H/2 x W/2
                # ---------------------------------------------------------------------
                block = ReluConv2D('d2_0', block, 64)
                block_start = block
                block = ReluConv2D('d2_1', block, 64)
                block = ReluConv2D('d2_2', block, 64)
                block = ReluConv2D('d2_3', block, 64, kernel_shape=1)
                block = tf.add(block_start, block, name='block_skip_B')
                skip_spatial_out.append(block)
                skip_unet_out.append(block)
                block = Merge(skip_spatial_in, 1, block, 'd24_s')

                # H/2 x W/2 -> H/4 x W/4
                # ---------------------------------------------------------------------
                block = ReluConv2D('d3_0', block, 128, stride=2)
                block = Merge(skip_temporal_in, 1, block, 'd31_s')
                block_start = block
                block = ReluConv2D('d3_1', block, 128)
                block = ReluConv2D('d3_2', block, 128)
                block = ReluConv2D('d3_3', block, 128, kernel_shape=1)
                block = tf.add(block_start, block, name='block_skip_C')
                skip_spatial_out.append(block)
                skip_unet_out.append(block)
                block = Merge(skip_spatial_in, 2, block, 'd34_s')

                # H/4 x W/4 -> H/8 x W/8
                # ---------------------------------------------------------------------
                block = ReluConv2D('d4_0', block, 256, stride=2)
                block = Merge(skip_temporal_in, 0, block, 'd41_s')
                block_start = block
                block = ReluConv2D('d4_1', block, 256)
                block = ReluConv2D('d4_2', block, 256)
                block = ReluConv2D('d4_3', block, 256, kernel_shape=1)
                block = tf.add(block_start, block, name='block_skip_D')
                skip_temporal_out.append(block)
                skip_spatial_out.append(block)
                block = Merge(skip_spatial_in, 3, block, 'd44_s')

                # H/8 x W/8 -> H/4 x W/4
                # ---------------------------------------------------------------------
                block = ReluDeconv2D('u1_0', block, 128, stride=2, kernel_shape=4)
                block = tf.add(block, skip_unet_out[1], name='skip01')
                block_start = block
                block = ReluConv2D('u1_1', block, 128)
                block = ReluConv2D('u1_2', block, 128)
                block = ReluConv2D('u1_3', block, 128)
                block = tf.add(block, block_start, name='block_skip_E')
                skip_temporal_out.append(block)
                skip_spatial_out.append(block)
                block = Merge(skip_spatial_in, 4, block, 'u14_s')

                # H/4 x W/4 -> H/2 x W/2
                # ---------------------------------------------------------------------
                block = ReluDeconv2D('u2_0', block, 64, stride=2, kernel_shape=4)
                block = tf.add(block, skip_unet_out[0], name='skip02')
                block_start = block
                block = ReluConv2D('u2_1', block, 64)
                block = ReluConv2D('u2_2', block, 64)
                block = ReluConv2D('u2_3', block, 64)
                block = tf.add(block, block_start, name='block_skip_F')
                skip_temporal_out.append(block)
                skip_spatial_out.append(block)
                block = Merge(skip_spatial_in, 5, block, 'u24_s')

                # H/2 x W/2 -> H x W
                # ---------------------------------------------------------------------
                block = ReluDeconv2D('u3_0', block, 64, stride=2, kernel_shape=4)
                block = ReluConv2D('u3_1', block, 64)
                block = ReluConv2D('u3_2', block, 64)
                block = ReluConv2D('u3_3', block, 6)
                block = ReluConv2D('u3_4', block, 3)
                estimate = tf.add(estimate, block, name='skip03')
                # skip_spatial_out.append(estimate)

                return estimate, skip_spatial_out, skip_temporal_out

    def _build_graph(self, input_vars):

        # some loss functions and metrics to track performance
        def l2_loss(x, y, name):
            return tf.reduce_mean(tf.squared_difference(x, y), name=name)

        def l1_loss(x, y, name):
            return tf.reduce_mean(tf.abs(x - y), name=name)

        def scaled_psnr(x, y, name):
            return symbf.psnr(128. * (x + 1.0), 128. * (y + 1.), 255, name=name)

        def image_pyramid(img, levels=LEVELS):
            """Resizing image to different shapes

            Args:
                img: image with original size
                levels (int, optional): number of resize steps

            Returns:
                images from small to original
            """
            with tf.name_scope('image_pyramid'):
                pyramid = [img]
                for i in range(levels - 1):
                    pyramid.append(resize_by_factor(img, 1. / (2**(i + 1))))
            return pyramid[::-1]

        # centered inputs [B, T, H, W, C]
        blurry, sharp = input_vars
        blurry = blurry / 128.0 - 1
        sharp = sharp / 128.0 - 1

        # take last as target
        expected_pyramid = image_pyramid(sharp[:, -1, :, :, :], levels=LEVELS)
        estimate_pyramid = image_pyramid(blurry[:, -1, :, :, :], levels=LEVELS)

        # track some performance metrics
        # never do dummy = [[]] * LEVELS
        l2err_list, l1err_list, psnr_list, psnr_impro_list = [], [], [], []
        for l in range(LEVELS):
            l2err_list.append([])
            l1err_list.append([])
            psnr_list.append([])
            psnr_impro_list.append([])
        # track the total costs for this model
        cost_list = []

        skip_spatial_out = [None] * LEVELS
        skip_temporal_out = [None] * LEVELS

        estimate_viz = []

        baseline_pyramid = image_pyramid(blurry[:, SEQ_LEN - 1, :, :, :], levels=LEVELS)
        psnr_base = [scaled_psnr(x, y, name="PSNR_base") for x, y in zip(baseline_pyramid, expected_pyramid)]

        for t in range(1, SEQ_LEN):
            logger.info("build time step: %i" % t)
            # get observation at all scales in time step 't'
            observation_pyramid = image_pyramid(blurry[:, SEQ_LEN - t - 1, :, :, :], levels=LEVELS)

            for l in range(LEVELS):
                ll = LEVELS - l - 1
                logger.info("level: {} with input shape {}".format(ll, observation_pyramid[l].get_shape()))
                # start with observation of smallest spatial size (l == 0)
                skip_spatial_in = None if (l == 0) else skip_spatial_out[l - 1]

                estimate_pyramid[l], skip_spatial_out[l], skip_temporal_out[l] = \
                    self.deblur_block(observation_pyramid[l],
                                      estimate_pyramid[l],
                                      skip_temporal_in=skip_temporal_out[l],
                                      skip_spatial_in=skip_spatial_in,
                                      name='level_%i_step_%i' % (ll, t))

                l2err_list[l].append(l2_loss(estimate_pyramid[l], expected_pyramid[l],
                                             name="L2loss_t%i_l%i" % (t, ll)))
                l1err_list[l].append(l1_loss(estimate_pyramid[l], expected_pyramid[l],
                                             name="L1loss_t%i_l%i" % (t, ll)))
                psnr_list[l].append(scaled_psnr(estimate_pyramid[l], expected_pyramid[l],
                                                name="PSNR_t%i_l%i" % (t, ll)))
                pi = tf.divide(psnr_list[l][-1], psnr_base[l], name="PSNR_IMPRO_t%i_l%i" % (t, ll))
                psnr_impro_list[l].append(pi)

                # we just optimize the costs on level 0
                # (otherwise we get artifacts as the CNN tends to focus on optimizing level > 0 only)
                if ll == 0:
                    cost_list.append(l2err_list[l][-1])

                # naming estimates for grabbing during deployment
                tf.identity((estimate_pyramid[l] + 1.0) * 128., name='estimate_t%i_l%i' % (t, ll))

                if(l == LEVELS - 1):
                    estimate_viz.append(estimate_pyramid[l])

        # just visualize original images
        with tf.name_scope('visualization'):

            expected = sharp[:, -1, :, :, :]
            estimate_viz = tf.concat(estimate_viz, axis=2)
            observed = tf.concat([blurry[:, i, :, :, :] for i in range(SEQ_LEN)], axis=2)

            viz = tf.concat([observed, estimate_viz, expected], axis=2, name='estimates')
            viz = 128.0 * (viz + 1.0)
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('blurry5_estimates5_expected', viz, max_outputs=max(30, BATCH_SIZE))

        # total cost is sum of all individual losses
        self.cost = tf.add_n(cost_list, name="total_cost")
        add_moving_summary(self.cost)

        for l in range(LEVELS):
            add_moving_summary(l2err_list[l] + l1err_list[l] + psnr_list[l] + psnr_impro_list[l])

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 0.005, summary=True)
        return tf.train.AdamOptimizer(lr)


def get_config(batch_size):
    logger.auto_set_dir('n')
    lmdbs = glob.glob('/graphics/projects/scratch/wieschol/YouTubeDataset/train*.lmdb')
    ds_train = [YoutubeData(lmdb, shape=(128, 128), ego_motion_size=17) for lmdb in lmdbs]
    ds_train = RandomMixData(ds_train)
    ds_train = BatchData(ds_train, BATCH_SIZE)
    ds_train = PrefetchDataZMQ(ds_train, 8)

    lmdbs = glob.glob('/graphics/projects/scratch/wieschol/YouTubeDataset/val*.lmdb')
    ds_val = [YoutubeData(lmdb, shape=(128, 128), ego_motion_size=17) for lmdb in lmdbs]
    ds_val = RandomMixData(ds_val)
    ds_val = BatchData(ds_val, BATCH_SIZE)
    ds_val = FixedSizeData(ds_val, 100)
    ds_val = PrefetchDataZMQ(ds_val, 8)

    steps_per_epoch = 1000

    return TrainConfig(dataflow=ds_train,
                       callbacks=[
                           ModelSaver(),
                           InferenceRunner(ds_val, [ScalarStats('total_cost'),
                                                    ScalarStats('PSNR_IMPRO_t%i_l0' % (SEQ_LEN - 1))])
                       ],
                       extra_callbacks=[
                           MovingAverageSummary(),
                           ProgressBar(['tower0/PSNR_base',
                                        'tower0/PSNR_IMPRO_t%i_l0' % (SEQ_LEN - 1),
                                        'tower0/PSNR_IMPRO_t%i_l1' % (SEQ_LEN - 1),
                                        'tower0/PSNR_IMPRO_t%i_l2' % (SEQ_LEN - 1),
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
    parser.add_argument('--load', help='load model')

    args = parser.parse_args()

    NR_GPU = len(args.gpu.split(','))
    with change_gpu(args.gpu):
        config = get_config(args.batch)
        if args.load:
            config.session_init = SaverRestore(args.load)
        config.nr_tower = NR_GPU
        SyncMultiGPUTrainer(config).train()
