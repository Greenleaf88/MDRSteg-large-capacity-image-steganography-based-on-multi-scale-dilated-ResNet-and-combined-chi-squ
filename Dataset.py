import pdb
import numpy as np
import tensorflow as tf
import random

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.data import Iterator
import matplotlib.pyplot as plt
import scipy

class SegDataLoader(object):
    def __init__(self, main_dir, batch_size, resize_shape, crop_shape, paths_file, buffer_size=100, split='train'):
        self.main_dir= main_dir
        self.batch_size= batch_size
        self.resize_shape= resize_shape
        self.crop_shape= crop_shape
        self.buffer_size= buffer_size
        self.paths_file= paths_file

        self.imgs_files= []
        self.labels_files= []

        # Read image and label paths from file and fill in self.images, self.labels
        self.parse_file(self.paths_file)

        if split == 'train':
            self.shuffle_lists()
        else:
            half_size = len(self.imgs_files) // 2
            self.imgs_files = self.imgs_files[:half_size]
            self.labels_files = self.labels_files[half_size:half_size*2]

        self.data_len= len(self.imgs_files)
        print('num of train: %d  num of valid: %d'%(len(self.imgs_files), len(self.labels_files)))

        img= convert_to_tensor(self.imgs_files, dtype= dtypes.string)
        label= convert_to_tensor(self.labels_files, dtype= dtypes.string)
        data_tr = tf.data.Dataset.from_tensor_slices((img, label))

        if split == 'train':
            data_tr = data_tr.map(self.parse_train, num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)
            data_tr = data_tr.shuffle(buffer_size)
        else:
            data_tr = data_tr.map(self.parse_val,num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)

        data_tr= data_tr.batch(batch_size)
        self.data_tr = data_tr.prefetch(buffer_size=self.batch_size)

    def shuffle_lists(self):
        # imgs= self.imgs_files
        # labels= self.labels_files
        #
        # permutation= np.random.permutation(len(self.imgs_files))
        # self.imgs_files= []
        # self.labels_files= []
        # for i in permutation:
        #     self.imgs_files.append(imgs[i])
        #     self.labels_files.append(labels[i])
        random.shuffle(self.imgs_files)
        random.shuffle(self.labels_files)

    def parse_train(self, im_path, label_path):
        # Load image
        img= tf.read_file(im_path)
        img= tf.image.decode_png(img, channels=3)
        # last_image_dim = tf.shape(img)[-1]

        # Load label
        label= tf.read_file(label_path)
        label= tf.image.decode_png(label, channels=1)

        # Scale
        # img = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.BICUBIC)
        # label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.BICUBIC)

#        # combine input and label
#        label = tf.cast(label, dtype=tf.float32)
#         combined = tf.concat([img, label], 2)
#
#        # flipping
#        combined= tf.image.random_flip_left_right(combined)
#
#        # cropping
        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = tf.random_crop(img,[self.crop_shape[0],self.crop_shape[1],3]) # TODO: Make cropping size a variable
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = tf.random_crop(label,[self.crop_shape[0],self.crop_shape[1],1])
        else:
            label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        # img_crop = tf.cast(img_crop, tf.float32)
        label_crop = tf.cast(label_crop, tf.float32) / 255.
        # label_crop = tf.cast(label_crop, tf.float32)

        # img, label = (combined_crop[:, :, :last_image_dim], combined_crop[:, :, last_image_dim:])
        # label = tf.cast(label, dtype=tf.uint8)
#        img.set_shape((self.crop_shape[0], self.crop_shape[1], 3))
#        label.set_shape((self.crop_shape[0], self.crop_shape[1], 1))
        return img_crop, label_crop

    def parse_val(self, im_path, label_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_png(img, channels=3)

        # Load label
        label = tf.read_file(label_path)
        label = tf.image.decode_png(label, channels=1)

        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = img[:self.crop_shape, :self.crop_shape, :]
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = label[:self.crop_shape, :self.crop_shape, :]
        else:
            label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        # img_crop = tf.cast(img_crop, tf.float32)
        label_crop = tf.cast(label_crop, tf.float32) / 255.
        # label_crop = tf.cast(label_crop, tf.float32)

        return img_crop, label_crop

    def parse_file(self, path):
        ff= open(path, 'r')
        for line in ff:
            tokens= line.strip().split(' ')
            self.imgs_files.append(self.main_dir+tokens[0])
            self.labels_files.append(self.main_dir+tokens[0])

    def print_files(self):
        for x, y in zip(self.imgs_files, self.labels_files):
            print(x, y)

class VocDataLoader(object):
    def __init__(self, main_dir, batch_size, resize_shape, crop_shape, paths_file, buffer_size=100, split='train'):
        self.main_dir= main_dir
        self.batch_size= batch_size
        self.resize_shape= resize_shape
        self.crop_shape= crop_shape
        self.buffer_size= buffer_size
        self.paths_file= paths_file

        self.imgs_files= []
        #self.labels_files= []

        # Read image and label paths from file and fill in self.images, self.labels
        self.parse_file(self.paths_file)

        if split == 'train':
            self.shuffle_lists()
        #else:
        #    half_size = len(self.imgs_files) // 2
        #    self.imgs_files = self.imgs_files[:half_size]
        #    self.labels_files = self.labels_files[half_size:half_size*2]

        self.data_len= len(self.imgs_files)
        #print('num of train: %d  num of valid: %d'%(len(self.imgs_files), len(self.labels_files)))

        img= convert_to_tensor(self.imgs_files, dtype= dtypes.string)
        #label= convert_to_tensor(self.labels_files, dtype= dtypes.string)
        #data_tr = tf.data.Dataset.from_tensor_slices((img, label))
        data_tr = tf.data.Dataset.from_tensor_slices((img))

        if split == 'train':
            data_tr = data_tr.map(self.parse_train, num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)
            data_tr = data_tr.shuffle(buffer_size)
        else:
            data_tr = data_tr.map(self.parse_val,num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)

        data_tr= data_tr.batch(batch_size)
        self.data_tr = data_tr.prefetch(buffer_size=self.batch_size)

    def shuffle_lists(self):
        # imgs= self.imgs_files
        # labels= self.labels_files
        #
        # permutation= np.random.permutation(len(self.imgs_files))
        # self.imgs_files= []
        # self.labels_files= []
        # for i in permutation:
        #     self.imgs_files.append(imgs[i])
        #     self.labels_files.append(labels[i])
        random.shuffle(self.imgs_files)
        #random.shuffle(self.labels_files)

    def parse_train(self, im_path):
        # Load image
        img= tf.read_file(im_path)
        img= tf.image.decode_png(img, channels=3)
        # last_image_dim = tf.shape(img)[-1]

        # Load label
        #label= tf.read_file(label_path)
        #label= tf.image.decode_png(label, channels=1)

        # Scale
        # img = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.BICUBIC)
        # label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.BICUBIC)

#        # combine input and label
#        label = tf.cast(label, dtype=tf.float32)
#         combined = tf.concat([img, label], 2)
#
#        # flipping
#        combined= tf.image.random_flip_left_right(combined)
#
#        # cropping
        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = tf.random_crop(img,[self.crop_shape[0],self.crop_shape[1],3]) # TODO: Make cropping size a variable
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        #if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
        #    label_crop = tf.random_crop(label,[self.crop_shape[0],self.crop_shape[1],1])
        #else:
        #    label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        # img_crop = tf.cast(img_crop, tf.float32)
        #label_crop = tf.cast(label_crop, tf.float32) / 255.
        # label_crop = tf.cast(label_crop, tf.float32)

        # img, label = (combined_crop[:, :, :last_image_dim], combined_crop[:, :, last_image_dim:])
        # label = tf.cast(label, dtype=tf.uint8)
#        img.set_shape((self.crop_shape[0], self.crop_shape[1], 3))
#        label.set_shape((self.crop_shape[0], self.crop_shape[1], 1))
        return img_crop

    def parse_val(self, im_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_png(img, channels=3)

        # Load label

        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = img[:self.crop_shape, :self.crop_shape, :]
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        img_crop = tf.cast(img_crop, tf.float32) / 255.
        # img_crop = tf.cast(img_crop, tf.float32)
        # label_crop = tf.cast(label_crop, tf.float32)

        return img_crop

    def parse_file(self, path):
        ff= open(path, 'r')
        for line in ff:
            tokens= line.strip().split(' ')
            self.imgs_files.append(self.main_dir+tokens[0])
            #self.labels_files.append(self.main_dir+tokens[0])

class VocRgbDataLoader(object):
    def __init__(self, main_dir, batch_size, resize_shape, crop_shape, paths_file, buffer_size=100, split='train'):
        self.main_dir= main_dir
        self.batch_size= batch_size
        self.resize_shape= resize_shape
        self.crop_shape= crop_shape
        self.buffer_size= buffer_size
        self.paths_file= paths_file

        self.imgs_files= []
        self.labels_files= []

        # Read image and label paths from file and fill in self.images, self.labels
        self.parse_file(self.paths_file)

        if split == 'train':
            self.shuffle_lists()
        else:
            half_size = len(self.imgs_files) // 2
            self.imgs_files = self.imgs_files[:half_size]
            self.labels_files = self.labels_files[half_size:half_size*2]

        self.data_len= len(self.imgs_files)
        print('num of train: %d  num of valid: %d'%(len(self.imgs_files), len(self.labels_files)))

        img= convert_to_tensor(self.imgs_files, dtype= dtypes.string)
        label= convert_to_tensor(self.labels_files, dtype= dtypes.string)
        data_tr = tf.data.Dataset.from_tensor_slices((img, label))

        if split == 'train':
            data_tr = data_tr.map(self.parse_train, num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)
            data_tr = data_tr.shuffle(buffer_size)
        else:
            data_tr = data_tr.map(self.parse_val,num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)

        data_tr= data_tr.batch(batch_size)
        self.data_tr = data_tr.prefetch(buffer_size=self.batch_size)

    def shuffle_lists(self):
        # imgs= self.imgs_files
        # labels= self.labels_files
        #
        # permutation= np.random.permutation(len(self.imgs_files))
        # self.imgs_files= []
        # self.labels_files= []
        # for i in permutation:
        #     self.imgs_files.append(imgs[i])
        #     self.labels_files.append(labels[i])
        random.shuffle(self.imgs_files)
        random.shuffle(self.labels_files)

    def parse_train(self, im_path, label_path):
        # Load image
        img= tf.read_file(im_path)
        img= tf.image.decode_png(img, channels=3)

        # Load label
        label= tf.read_file(label_path)
        label= tf.image.decode_png(label, channels=3)

        # Scale
        # img = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.BICUBIC)
        # label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # label = tf.image.resize_images(label, self.resize_shape, method= tf.image.ResizeMethod.BICUBIC)

#        # combine input and label
#        label = tf.cast(label, dtype=tf.float32)
#         combined = tf.concat([img, label], 2)
#
#        # flipping
#        combined= tf.image.random_flip_left_right(combined)
#
#        # cropping
        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = tf.random_crop(img,[self.crop_shape[0],self.crop_shape[1],3])
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = tf.random_crop(label,[self.crop_shape[0],self.crop_shape[1],1])
        else:
            label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        label_crop = tf.cast(label_crop, tf.float32) / 255.

        return img_crop, label_crop

    def parse_val(self, im_path, label_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_png(img, channels=3)

        # Load label
        label = tf.read_file(label_path)
        label = tf.image.decode_png(label, channels=3)

        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = img[:self.crop_shape, :self.crop_shape, :]
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = label[:self.crop_shape, :self.crop_shape, :]
        else:
            label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        label_crop = tf.cast(label_crop, tf.float32) / 255.

        return img_crop, label_crop

    def parse_file(self, path):
        ff= open(path, 'r')
        for line in ff:
            tokens= line.strip().split(' ')
            self.imgs_files.append(self.main_dir+tokens[0])
            self.labels_files.append(self.main_dir+tokens[0])

    def print_files(self):
        for x, y in zip(self.imgs_files, self.labels_files):
            print(x, y)

class LfwRgbDataLoader(object):
    def __init__(self, main_dir, batch_size, resize_shape, crop_shape, paths_file, buffer_size=100, split='train'):
        self.main_dir= main_dir
        self.batch_size= batch_size
        self.resize_shape= resize_shape
        self.crop_shape= crop_shape
        self.buffer_size= buffer_size
        self.paths_file= paths_file

        self.imgs_files= []
        self.labels_files= []

        # Read image and label paths from file and fill in self.images, self.labels
        self.parse_file(self.paths_file)

        if split == 'train':
            self.shuffle_lists()
        else:
            half_size = len(self.imgs_files) // 2
            self.imgs_files = self.imgs_files[:half_size]
            self.labels_files = self.labels_files[half_size:half_size*2]

        self.data_len= len(self.imgs_files)
        print('num of train: %d  num of valid: %d'%(len(self.imgs_files), len(self.labels_files)))

        img= convert_to_tensor(self.imgs_files, dtype= dtypes.string)
        label= convert_to_tensor(self.labels_files, dtype= dtypes.string)
        data_tr = tf.data.Dataset.from_tensor_slices((img, label))

        if split == 'train':
            data_tr = data_tr.map(self.parse_train, num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)
            data_tr = data_tr.shuffle(buffer_size)
        else:
            data_tr = data_tr.map(self.parse_val,num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)

        data_tr= data_tr.batch(batch_size)
        self.data_tr = data_tr.prefetch(buffer_size=self.batch_size)

    def shuffle_lists(self):
        random.shuffle(self.imgs_files)
        random.shuffle(self.labels_files)

    def parse_train(self, im_path, label_path):
        # Load image
        img= tf.read_file(im_path)
        img= tf.image.decode_png(img, channels=3)

        # Load label
        label= tf.read_file(label_path)
        label= tf.image.decode_png(label, channels=3)
#
#        # flipping
#        combined= tf.image.random_flip_left_right(combined)
#
#        # cropping
        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = tf.random_crop(img,[self.crop_shape[0],self.crop_shape[1],3])
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = tf.random_crop(label,[self.crop_shape[0],self.crop_shape[1],1])
        else:
            label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        label_crop = tf.cast(label_crop, tf.float32) / 255.

        return img_crop, label_crop

    def parse_val(self, im_path, label_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_png(img, channels=3)

        # Load label
        label = tf.read_file(label_path)
        label = tf.image.decode_png(label, channels=3)

        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = img[:self.crop_shape, :self.crop_shape, :]
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = label[:self.crop_shape, :self.crop_shape, :]
        else:
            label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        label_crop = tf.cast(label_crop, tf.float32) / 255.

        return img_crop, label_crop

    def parse_file(self, path):
        ff= open(path, 'r')
        for line in ff:
            tokens= line.strip().split(' ')
            self.imgs_files.append(self.main_dir+tokens[0])
            self.labels_files.append(self.main_dir+tokens[0])

    def print_files(self):
        for x, y in zip(self.imgs_files, self.labels_files):
            print(x, y)

class ImageNetRgbDataLoader(object):
    def __init__(self, main_dir, batch_size, resize_shape, crop_shape, paths_file, buffer_size=100, split='train'):
        self.main_dir= main_dir
        self.batch_size= batch_size
        self.resize_shape= resize_shape
        self.crop_shape= crop_shape
        self.buffer_size= buffer_size
        self.paths_file= paths_file

        self.imgs_files= []
        self.labels_files= []

        # Read image and label paths from file and fill in self.images, self.labels
        self.parse_file(self.paths_file)

        if split == 'train':
            self.shuffle_lists()
        else:
            half_size = len(self.imgs_files) // 2
            self.imgs_files = self.imgs_files[:half_size]
            self.labels_files = self.labels_files[half_size:half_size*2]

        self.data_len= len(self.imgs_files)
        print('num of train: %d  num of valid: %d'%(len(self.imgs_files), len(self.labels_files)))

        img= convert_to_tensor(self.imgs_files, dtype= dtypes.string)
        label= convert_to_tensor(self.labels_files, dtype= dtypes.string)
        data_tr = tf.data.Dataset.from_tensor_slices((img, label))

        if split == 'train':
            data_tr = data_tr.map(self.parse_train, num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)
            data_tr = data_tr.shuffle(buffer_size)
        else:
            data_tr = data_tr.map(self.parse_val,num_parallel_calls=8)#, num_threads=8, output_buffer_size=100*self.batch_size)

        data_tr= data_tr.batch(batch_size)
        self.data_tr = data_tr.prefetch(buffer_size=self.batch_size)

    def shuffle_lists(self):
        random.shuffle(self.imgs_files)
        random.shuffle(self.labels_files)

    def parse_train(self, im_path, label_path):
        # Load image
        img= tf.read_file(im_path)
        img= tf.image.decode_png(img, channels=3)

        # Load label
        label= tf.read_file(label_path)
        label= tf.image.decode_png(label, channels=3)
#
#        # flipping
#        combined= tf.image.random_flip_left_right(combined)
#
#        # cropping
        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = tf.random_crop(img,[self.crop_shape[0],self.crop_shape[1],3])
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = tf.random_crop(label,[self.crop_shape[0],self.crop_shape[1],1])
        else:
            label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        label_crop = tf.cast(label_crop, tf.float32) / 255.

        return img_crop, label_crop

    def parse_val(self, im_path, label_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_png(img, channels=3)

        # Load label
        label = tf.read_file(label_path)
        label = tf.image.decode_png(label, channels=3)

        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = img[:self.crop_shape, :self.crop_shape, :]
        else:
            img_crop = tf.image.resize_images(img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = label[:self.crop_shape, :self.crop_shape, :]
        else:
            label_crop = tf.image.resize_images(label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        label_crop = tf.cast(label_crop, tf.float32) / 255.

        return img_crop, label_crop

    def parse_file(self, path):
        ff= open(path, 'r')
        for line in ff:
            tokens= line.strip().split(' ')
            self.imgs_files.append(self.main_dir+tokens[0])
            self.labels_files.append(self.main_dir+tokens[0])

    def print_files(self):
        for x, y in zip(self.imgs_files, self.labels_files):
            print(x, y)

if __name__=="__main__":

    config= tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session= tf.Session(config=config)

    with tf.device('/cpu:0'):
        # segdl= SegDataLoader('/home/moliq/Documents/VOC2012/JPEGImages/', 2, (256,256), (256,256), 'voc_train.txt', split='train')
        # segdl= LfwRgbDataLoader('/home/moliq/Documents/lfw/', 2, (256,256), (256,256), 'dataset/lfw_train.txt', split='train')
        segdl= ImageNetRgbDataLoader('/home/moliq/Documents/imagenet/ILSVRC2012_img_val/', 2, (256,256), (256,256), 'dataset/imagenet_train.txt', split='train')
        # segdl= SegDataLoader('/home/eren/Data/Cityscapes/', 10, (512,1024), (512,512), 'val.txt', split='val')
        iterator = Iterator.from_structure(segdl.data_tr.output_types, segdl.data_tr.output_shapes)
        next_batch= iterator.get_next()

        training_init_op = iterator.make_initializer(segdl.data_tr)
        session.run(training_init_op)

    steps_per_epoch = segdl.data_len // segdl.batch_size

    for epoch in range(1, 5):
        print('epoch %d'%epoch)
        segdl.shuffle_lists()
        session.run(training_init_op)
        for i in range(steps_per_epoch):
            img_batch, label_batch = session.run(next_batch)
            print(img_batch.mean(), img_batch.std())
            print(i)

    # for i in range(10):
    #     img_batch, label_batch = session.run(next_batch)
    #     print(img_batch)
#       img_batch= np.asarray(img_batch,dtype=np.uint8)
#       plt.imshow(label_batch[0,0,:,:,0]);plt.show()
#       plt.imshow(img_batch[0,0,:,:,:]);plt.show()


