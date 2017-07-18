
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import argparse
import tensorflow as tf
import numpy as np
from data_provider import get_data as YoutubeData
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbf
import glob

"""
Learning Blind Motion Deblurring
"""

SEQ_LEN = 5
BATCH_SIZE = 32


def ReluConv2D(name, x, out_channels, use_relu=True, kernel_shape=3, stride=1):
    if use_relu:
        x = tf.nn.relu(x, name='%s_relu' % name)
    x = Conv2D('%s_conv' % name, x, out_channels, kernel_shape=kernel_shape, stride=stride)
    return x


def ReluDeconv2D(name, x, out_channels, kernel_shape=3, stride=1):
    x = tf.nn.relu(x, name='%s_relu' % name)
    x = Deconv2D('%s_deconv' % name, x, out_channels, kernel_shape=kernel_shape, stride=stride)
    return x


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SEQ_LEN, None, None, 3), 'blurry'),
                InputDesc(tf.float32, (None, SEQ_LEN, None, None, 3), 'sharp')]

    @auto_reuse_variable_scope
    def deblur_block(self, observation, estimate, block_id=0, skip=None):
        """apply one deblur step
        Args:
            observation: new unseen observation
            estimate: latest estimate
            block_id (int, optional): running id in recurrent structure
            skip (None, optional): list of skip_connections
        """
        logger.info("create deblur_block %i" % block_id)
        with tf.name_scope("deblur_block"):
            with argscope(BatchNorm, use_local_stat=True), \
                    argscope([Conv2D, Deconv2D], nl=lambda x, name: BatchNorm(name, x)):
                inputs = tf.concat([observation, estimate], 3)
                d02 = ReluConv2D('d0', inputs, 32, stride=1, kernel_shape=3)

                # H x W -> H/2 x W/2
                d11 = ReluConv2D('d1_0', d02, 64, stride=2)
                d12 = ReluConv2D('d1_1', d11, 64)
                d13 = ReluConv2D('d1_2', d12, 64)
                d14 = ReluConv2D('d1_3', d13, 64, kernel_shape=1)
                d14 = tf.add(d11, d14, name='block_skip_A')

                # H/2 x W/2 -> H/2 x W/2 (dilated here?)
                d21 = ReluConv2D('d2_0', d14, 64)
                d22 = ReluConv2D('d2_1', d21, 64)
                d23 = ReluConv2D('d2_2', d22, 64)
                d24 = ReluConv2D('d2_3', d23, 64, kernel_shape=1)
                d24 = tf.add(d21, d24, name='block_skip_B')

                # H/2 x W/2 -> H/4 x W/4
                d31 = ReluConv2D('d3_0', d21, 128, stride=2)
                d32 = ReluConv2D('d3_1', d31, 128)
                d33 = ReluConv2D('d3_2', d32, 128)
                d34 = ReluConv2D('d3_3', d33, 128, kernel_shape=1)
                d34 = tf.add(d31, d34, name='block_skip_C')

                # H/4 x W/4 -> H/8 x W/8
                d41 = ReluConv2D('d4_0', d31, 256, stride=2)
                if len(skip) == 0:
                    skip.append(tf.zeros_like(d41))
                # -- begin temporal skip -----------
                skip[0] = tf.concat([d41, skip[0]], axis=3)
                skip[0] = ReluConv2D('s1', skip[0], 256, kernel_shape=1)
                if block_id > 1:
                    d41 = skip[0]
                # -- end temporal skip   -----------
                d42 = ReluConv2D('d4_1', d41, 256)
                d43 = ReluConv2D('d4_2', d42, 256)
                d44 = ReluConv2D('d4_3', d43, 256, kernel_shape=1)
                skip[0] = ReluConv2D('so1', d44, 256)
                d44 = tf.add(d41, d44, name='block_skip_D')

                # H/8 x W/8 -> H/4 x W/4
                u11 = ReluDeconv2D('u1_0', d44, 128, stride=2, kernel_shape=4)

                print 'u11', u11.get_shape()
                print 'd34', d44.get_shape()
                u11 = tf.add(u11, d34, name='skip01')
                u12 = ReluConv2D('u1_1', u11, 128)
                if len(skip) == 1:
                    skip.append(tf.zeros_like(u12))
                # -- begin temporal skip -----------
                skip[1] = tf.concat([u12, skip[1]], axis=3)
                skip[1] = ReluConv2D('s2', skip[1], 128, kernel_shape=1)
                if block_id > 1:
                    u12 = skip[1]
                # -- end temporal skip   -----------
                u13 = ReluConv2D('u1_2', u12, 128)
                skip[1] = ReluConv2D('so2', u13, 128)
                u14 = ReluConv2D('u1_3', u13, 128)
                u14 = tf.add(u14, u11, name='block_skip_E')

                # H/4 x W/4 -> H/2 x W/2
                u21 = ReluDeconv2D('u2_0', u14, 64, stride=2, kernel_shape=4)
                u21 = tf.add(u21, d24, name='skip02')
                u22 = ReluConv2D('u2_1', u21, 64)
                if len(skip) == 2:
                    skip.append(tf.zeros_like(u22))
                # -- begin temporal skip -----------
                skip[2] = tf.concat([u22, skip[2]], axis=3)
                skip[2] = ReluConv2D('s3', skip[2], 64, kernel_shape=1)
                if block_id > 1:
                    u22 = skip[2]
                # -- end temporal skip   -----------
                u23 = ReluConv2D('u2_2', u22, 64)
                skip[2] = ReluConv2D('so32', u23, 64)
                u24 = ReluConv2D('u2_3', u23, 64)
                u24 = tf.add(u24, u21, name='block_skip_F')

                # H/2 x W/2 -> H x W
                u31 = ReluDeconv2D('u3_0', u24, 64, stride=2, kernel_shape=4)
                u32 = ReluConv2D('u3_1', u31, 64)
                u33 = ReluConv2D('u3_2', u32, 64)
                u34 = ReluConv2D('u3_3', u33, 6)
                u35 = ReluConv2D('u3_4', u34, 3)
                estimate = tf.add(estimate, u35, name='skip03')

                return estimate, skip

    def _build_graph(self, input_vars):

        # centered inputs [B, T, H, W, C]
        blurry, sharp = input_vars
        blurry = blurry / 128.0 - 1
        sharp = sharp / 128.0 - 1

        # take last as target
        expected = sharp[:, -1, :, :, :]
        estimate = blurry[:, -1, :, :, :]

        l2err_list, l1err_list, psnr_list, psnr_impro_list = [], [], [], []

        skip = []
        # skip.append(tf.constant(np.zeros((BATCH_SIZE, 16, 16, 256), dtype=np.float32)))
        # skip.append(tf.constant(np.zeros((BATCH_SIZE, 32, 32, 128), dtype=np.float32)))
        # skip.append(tf.constant(np.zeros((BATCH_SIZE, 64, 64, 64), dtype=np.float32)))

        psnr_i = symbf.psnr(128. * (estimate + 1.0), 128. * (expected + 1.), 255, name="psnr_0")
        psnr_list.append(psnr_i)

        estimates = []
        for t in range(1, SEQ_LEN):
            observation = blurry[:, -1 - t, :, :, :]
            estimate, skip = self.deblur_block(observation, estimate, t, skip=skip)
            estimates.append(estimate)

            # tracking losses
            l2err_list.append(tf.reduce_mean(tf.squared_difference(estimate, expected), name="L2loss_block_%i" % t))
            l1err_list.append(tf.reduce_mean(tf.abs(estimate - expected), name='L1loss_block%i' % t))

            # tracking psnr
            psnr = symbf.psnr(128. * (estimate + 1.0), 128. * (expected + 1.), 255, name="psnr_%i" % t)
            psnr_list.append(psnr)
            psnr_impro_list.append(tf.divide(psnr, psnr_i, name="psnr_improv_%i" % t))

            # naming estimates for grabbing during deployment
            tf.identity((estimate + 1.0) * 128., name='estimate_%i' % t)

        with tf.name_scope('visualization'):
            viz = tf.unstack(blurry, num=SEQ_LEN, axis=1) + estimates + [expected]
            viz = 128.0 * (tf.concat(viz, axis=2, name='estimates') + 1.0)
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('blurry5_estimates5_expected', viz, max_outputs=max(30, BATCH_SIZE))

        # total cost is sum of all individual losses
        self.cost = tf.add_n(l2err_list, name="total_cost")
        add_moving_summary([self.costdil01] + l2err_list + l1err_list + psnr_list + psnr_impro_list)

        with tf.name_scope('histograms'):
            tf.summary.histogram('l2err_list', tf.stack([tf.expand_dims(d, -1) for d in l2err_list], axis=1))
            tf.summary.histogram('l1err_list', tf.stack([tf.expand_dims(d, -1) for d in l1err_list], axis=1))
            tf.summary.histogram('psnr_list', tf.stack([tf.expand_dims(d, -1) for d in psnr_list], axis=1))
            tf.summary.histogram('psnr_impro_list', tf.stack([tf.expand_dims(d, -1) for d in psnr_impro_list], axis=1))

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 0.005, summary=True)
        return tf.train.AdamOptimizer(lr)


def get_config(batch_size):
    logger.auto_set_dir()
    lmdbs = glob.glob('/graphics/projects/scratch/wieschol/YouTubeDataset/train*.lmdb')
    ds = [YoutubeData(lmdb, shape=(128, 128), ego_motion_size=[17, 25, 35, 71]) for lmdb in lmdbs]
    dataset_train = RandomMixData(ds)
    dataset_train = BatchData(dataset_train, BATCH_SIZE)
    dataset_train = PrefetchDataZMQ(dataset_train, 8)

    steps_per_epoch = 1000

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(['tower0/psnr_%i' % (SEQ_LEN - 1), 'tower0/psnr_0',
                         'tower0/psnr_improv_%i' % (SEQ_LEN - 1)]),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=400,
    )


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
