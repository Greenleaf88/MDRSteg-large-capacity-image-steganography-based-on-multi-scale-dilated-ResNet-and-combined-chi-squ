import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error


class WeightedSmoothL1LocalizationLoss():
  """Smooth L1 localization loss function aka Huber Loss..
  The smooth L1_loss is defined elementwise as .5 x^2 if |x| <= delta and
  delta * (|x|- 0.5*delta) otherwise, where x is the difference between
  predictions and target.
  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """

  def __init__(self, delta=1.0):
    """Constructor.
    Args:
      delta: delta for smooth L1 loss.
    """
    self._delta = delta

  def compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.
    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    return tf.reduce_sum(tf.losses.huber_loss(
        target_tensor,
        prediction_tensor,
        delta=self._delta,
        weights=tf.expand_dims(weights, axis=2),
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE
    ), axis=2)

def rgb2yuv_tf(image):
    Y = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
    U = - 0.14713 * image[:, :, :, 0] - 0.28886 * image[:, :, :, 1] + 0.436 * image[:, :, :, 2]
    V = 0.615 * image[:, :, :, 0] - 0.51449 * image[:, :, :, 1] - 0.10001 * image[:, :, :, 2]
    return tf.stack([Y, U, V], axis=-1)

def yuv2rgb_tf(image):
    R = image[:,:, :, 0] + 1.13983 * image[:,:, :, 2]
    G = image[:,:, :, 0] - 0.39465 * image[:,:, :, 1] - 0.58060 * image[:,:, :, 2]
    B = image[:,:, :, 0] + 2.03211 * image[:,:, :, 1]
    return tf.stack([R, G, B], axis=-1)


def rgb2yuv_np(image):
    Y = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
    U = - 0.14713 * image[:, :, :, 0] - 0.28886 * image[:, :, :, 1] + 0.436 * image[:, :, :, 2]
    V = 0.615 * image[:, :, :, 0] - 0.51449 * image[:, :, :, 1] - 0.10001 * image[:, :, :, 2]
    return np.stack([Y, U, V], axis=-1)

def yuv2rgb_np(image):
    R = image[:, :, :, 0] + 1.13983 * image[:, :, :, 2]
    G = image[:, :, :, 0] - 0.39465 * image[:, :, :, 1] - 0.58060 * image[:, :, :, 2]
    B = image[:, :, :, 0] + 2.03211 * image[:, :, :, 1]
    return np.stack([R, G, B], axis=-1)

def mean_square_error(a, b):
    diff = a - b
    diff = diff * diff
    # diff = np.sqrt(diff)
    diff = np.mean(diff)
    return diff

def to_image(image):
    image = (((image - image.min()) * 255) / (image.max() - image.min())).astype(np.uint8)
    return image


def rgb2ycbcr(image):
    Y = 0 + 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    CB = 128.0 / 255 - 0.168736 * image[:, :, 0] - 0.331264 * image[:, :, 1] + 0.5 * image[:, :, 2]
    CR = 128.0 / 255 + 0.5 * image[:, :, 0] - 0.418688 * image[:, :, 1] - 0.081312 * image[:, :, 2]
    return np.stack([Y, CB, CR], axis=-1)

def rgb2yuv(image):
    Y = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    U = - 0.14713 * image[:, :, 0] - 0.28886 * image[:, :, 1] + 0.436 * image[:, :, 2]
    V = 0.615 * image[:, :, 0] - 0.51449 * image[:, :, 1] - 1.0001 * image[:, :, 2]
    return np.stack([Y, U, V], axis=-1)


def mse_test(c, d):
    red = mean_square_error(c[:, :, 0], d[:, :, 0])
    green = mean_square_error(c[:, :, 1], d[:, :, 1])
    blue = mean_square_error(c[:, :, 2], d[:, :, 2])
    sep = (red + green + blue) / 3
    combin = mean_square_error(c, d)
    return combin, sep

def yuv_vs_rgb(a, b):
    rgb = mean_square_error(a, b)
    a_hsv = rgb2yuv(a)
    b_hsv = rgb2yuv(b)
    hsv = mean_square_error(a_hsv, b_hsv)
    return rgb, hsv

def ColorDistance(rgb1,rgb2):
    '''d = {} distance between two colors(3)'''
    rm = 0.5*(rgb1[0]+rgb2[0])
    d = sum((2+rm,4,3-rm)*(rgb1-rgb2)**2)**0.5
    return d


def main():
    rgb1 = np.array([1, 1, 0])
    rgb2 = np.array([0, 0, 0])
    ColorDistance(rgb1, rgb2)

    cover = np.load('0 0.799 0.013 0.002 0.136 0.051_cover.npy')
    secret = np.load('0 0.799 0.013 0.002 0.136 0.051_secret.npy')
    secret_reveal = np.load('0 0.799 0.013 0.002 0.136 0.051_secret_reveal.npy')
    stego = np.load('0 0.799 0.013 0.002 0.136 0.051_stego.npy')
    a = to_image(cover[0])
    b = to_image(stego[0])
    c = to_image(secret[0])
    d = to_image(secret_reveal[0])
    print(yuv_vs_rgb(a, b))
    print(yuv_vs_rgb(c, d))
    mse = mean_square_error(cover[0], stego[0])
    mse2 = mean_square_error(secret[0], secret_reveal[0])
    print(mse, mse2)


if __name__ == '__main__':
    main()