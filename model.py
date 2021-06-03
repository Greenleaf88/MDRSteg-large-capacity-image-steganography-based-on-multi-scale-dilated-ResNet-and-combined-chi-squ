import os
import time
import numpy as np

import tensorflow as tf
from tensorflow.python.layers.convolutional import conv2d, separable_conv2d
from tensorflow.python.layers.normalization import batch_norm
from tensorflow.python.layers.pooling import max_pool2d
from tensorflow.python.layers.core import flatten
from tensorflow.python.layers.core import dense




def Discriminator(input, is_training=True, reusing=False, name=None):

    if name:
        name = 'Discriminator' + '_' + name
    else:
        name = 'Discriminator'
    with tf.variable_scope(name, reuse=reusing):

        with tf.variable_scope("conv1"):
            conv1 = conv2d(inputs=input, filters=16, kernel_size=3, padding='same', name="1",
                           activation=tf.nn.relu)
            conv1 = batch_norm(inputs=conv1, training=is_training, name='1')
            conv1 = conv2d(inputs=conv1, filters=16, kernel_size=3, strides=2, padding='same', name="2", activation=tf.nn.relu)
            conv1 = batch_norm(inputs=conv1, training=is_training, name='2')

        with tf.variable_scope("conv2"):
            conv2 = conv2d(inputs=conv1, filters=32, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv2 = batch_norm(inputs=conv2, training=is_training, name='1')
            conv2 = conv2d(inputs=conv2, filters=32, kernel_size=3, strides=2, padding='same', name="2", activation=tf.nn.relu)
            conv2 = batch_norm(inputs=conv2, training=is_training, name='2')

        with tf.variable_scope("conv3"):
            conv3 = conv2d(inputs=conv2, filters=64, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv3 = batch_norm(inputs=conv3, training=is_training, name='1')
            conv3 = conv2d(inputs=conv3, filters=64, kernel_size=3, strides=2, padding='same', name="2", activation=tf.nn.relu)
            conv3 = batch_norm(inputs=conv3, training=is_training, name='2')

        with tf.variable_scope("conv4"):
            conv4 = conv2d(inputs=conv3, filters=128, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv4 = batch_norm(inputs=conv4, training=is_training, name='1')
            conv4 = conv2d(inputs=conv4, filters=128, kernel_size=3, strides=2, padding='same', name="2", activation=tf.nn.relu)
            conv4 = batch_norm(inputs=conv4, training=is_training, name='2')

        with tf.variable_scope("conv5"):
            conv5 = conv2d(inputs=conv4, filters=128, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv5 = batch_norm(inputs=conv5, training=is_training, name='1')
            conv5 = conv2d(inputs=conv5, filters=128, kernel_size=3, strides=2, padding='same', name="2", activation=tf.nn.relu)
            conv5 = batch_norm(inputs=conv5, training=is_training, name='2')

        with tf.variable_scope("output"):
            conv5 = max_pool2d(inputs=conv5, pool_size=8, strides=1, name='global_max_pool')
            logits = flatten(conv5, name='flatten')
            logits = dense(inputs=logits, units=1,use_bias=False)


    return logits


def encode_net(Y, secret_tensor, is_training):
    with tf.variable_scope('hide_net'):
        concat_input = tf.concat([Y, secret_tensor], axis=-1, name='images_features_concat')

        with tf.variable_scope("conv1"):
            conv1 = conv2d(inputs=concat_input, filters=32, kernel_size=3, padding='same', name="1",
                           activation=tf.nn.relu)
            conv1 = batch_norm(inputs=conv1, training=is_training, name='1')
            conv1 = conv2d(inputs=conv1, filters=32, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv1 = batch_norm(inputs=conv1, training=is_training, name='2')

        with tf.variable_scope("conv2"):
            conv2 = conv2d(inputs=conv1, filters=64, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv2 = batch_norm(inputs=conv2, training=is_training, name='1')
            conv2 = conv2d(inputs=conv2, filters=64, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv2 = batch_norm(inputs=conv2, training=is_training, name='2')

        with tf.variable_scope("conv3"):
            conv3 = conv2d(inputs=conv2, filters=128, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv3 = batch_norm(inputs=conv3, training=is_training, name='1')
            conv3 = conv2d(inputs=conv3, filters=128, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv3 = batch_norm(inputs=conv3, training=is_training, name='2')

        with tf.variable_scope("conv4"):
            conv4 = conv2d(inputs=conv3, filters=128, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv4 = batch_norm(inputs=conv4, training=is_training, name='1')
            conv4 = conv2d(inputs=conv4, filters=128, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv4 = batch_norm(inputs=conv4, training=is_training, name='2')

        with tf.variable_scope("conv5"):
            conv5 = conv2d(inputs=conv4, filters=64, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv5 = batch_norm(inputs=conv5, training=is_training, name='1')
            conv5 = conv2d(inputs=conv5, filters=64, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv5 = batch_norm(inputs=conv5, training=is_training, name='2')

        with tf.variable_scope("conv6"):
            conv6 = conv2d(inputs=conv5, filters=32, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv6 = batch_norm(inputs=conv6, training=is_training, name='1')
            conv6 = conv2d(inputs=conv6, filters=32, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv6 = batch_norm(inputs=conv6, training=is_training, name='2')

        with tf.variable_scope("output"):
            y_output = conv2d(inputs=conv6, filters=1, kernel_size=1, padding='same', name='output')

    return y_output


def decode_net(container_tensor, is_training):
    with tf.variable_scope('reveal_net'):
        with tf.variable_scope("conv1"):
            conv1 = conv2d(inputs=container_tensor, filters=32, kernel_size=3, padding='same', name="1",
                           activation=tf.nn.relu)
            conv1 = batch_norm(inputs=conv1, training=is_training, name='1')
            conv1 = conv2d(inputs=conv1, filters=32, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv1 = batch_norm(inputs=conv1, training=is_training, name='2')

        with tf.variable_scope("conv2"):
            conv2 = conv2d(inputs=conv1, filters=64, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv2 = batch_norm(inputs=conv2, training=is_training, name='1')
            conv2 = conv2d(inputs=conv2, filters=64, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv2 = batch_norm(inputs=conv2, training=is_training, name='2')

        with tf.variable_scope("conv3"):
            conv3 = conv2d(inputs=conv2, filters=128, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv3 = batch_norm(inputs=conv3, training=is_training, name='1')
            conv3 = conv2d(inputs=conv3, filters=128, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv3 = batch_norm(inputs=conv3, training=is_training, name='2')

        with tf.variable_scope("conv4"):
            conv4 = conv2d(inputs=conv3, filters=128, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv4 = batch_norm(inputs=conv4, training=is_training, name='1')
            conv4 = conv2d(inputs=conv4, filters=128, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv4 = batch_norm(inputs=conv4, training=is_training, name='2')

        with tf.variable_scope("conv5"):
            conv5 = conv2d(inputs=conv4, filters=64, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv5 = batch_norm(inputs=conv5, training=is_training, name='1')
            conv5 = conv2d(inputs=conv5, filters=64, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv5 = batch_norm(inputs=conv5, training=is_training, name='2')

        with tf.variable_scope("conv6"):
            conv6 = conv2d(inputs=conv5, filters=32, kernel_size=3, padding='same', name="1", activation=tf.nn.relu)
            conv6 = batch_norm(inputs=conv6, training=is_training, name='1')
            conv6 = conv2d(inputs=conv6, filters=32, kernel_size=3, padding='same', name="2", activation=tf.nn.relu)
            conv6 = batch_norm(inputs=conv6, training=is_training, name='2')

        with tf.variable_scope("output"):
            output = conv2d(inputs=conv6, filters=1, kernel_size=1, padding='same', name='output')

        return output