# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
# import cv2

import tensorflow as tf
from tensorflow.data import Iterator

from Dataset import SegDataLoader, VocRgbDataLoader, VocDataLoader, LfwRgbDataLoader, ImageNetRgbDataLoader
from visulize import save_test_images
from utils import rgb2yuv_tf, yuv2rgb_tf
from model import Discriminator, encode_net, decode_net
from ResNet import resnet_nopooling


class Model():

    def __init__(self):
        self.run_time = time.strftime("%m%d-%H%M")
        # self.learning_rate = 0.0001
        self.starter_learning_rate = 0.001

        self.epoches = 70
        self.log_path = 'logs/'+self.run_time + '/'


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.InteractiveSession(config=config)
        self.secret_tensor = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32, name="secret_tensor")
        self.cover_tensor = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32, name="cover_tensor")

        self.cover_yuv = rgb2yuv_tf(self.cover_tensor)
        self.secret_yuv = rgb2yuv_tf(self.secret_tensor)

        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        # self.test_op = self.prepare_test_graph(self.secret_tensor, self.cover_tensor)

    def get_hiding_network_op(self, cover_tensor, secret_tensor, is_training):

        concat_input = tf.concat([cover_tensor, secret_tensor], axis=-1, name='images_features_concat')
        # output = resnet_nopooling(concat_input, name='encode', n_class=3, dilate=[2,4,8,16], is_training=is_training)
        output = resnet_nopooling(concat_input, name='encode', n_class=3, is_training=is_training)

        return output


    def get_reveal_network_op(self, container_tensor, is_training):

        output = resnet_nopooling(container_tensor, name='decode', n_class=3,  is_training=is_training)
        return output

    def get_noise_layer_op(self,tensor,std=.1):
        # with tf.variable_scope("noise_layer"):
        #     return tensor + tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=std, dtype=tf.float32)
        return tensor

    def get_loss_op(self,secret_true,secret_pred,cover_true,cover_pred):
        # D_real_secret = Discriminator(secret_true)
        # D_fake_secret = Discriminator(secret_pred, reusing=True)
        # D_real = Discriminator(cover_true, reusing=True)
        # D_fake = Discriminator(cover_pred, reusing=True)

        # D_real_secret = Discriminator(secret_true, name='secret', reusing=False)
        # D_fake_secret = Discriminator(secret_pred, name='secret', reusing=True)
        # D_real = Discriminator(cover_true, name='cover', reusing=False)
        # D_fake = Discriminator(cover_pred, name='cover', reusing=True)
        #
        # D_real = tf.concat([D_real, D_real_secret], axis=0, name='gan_true_concat')
        # D_fake = tf.concat([D_fake, D_fake_secret], axis=0, name='gan_pred_concat')
        #
        # D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        # G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        with tf.variable_scope("huber_losses"):
            # secret_mse = tf.losses.mean_squared_error(secret_true,secret_pred)
            # cover_mse = tf.losses.mean_squared_error(cover_true,cover_pred)
            # secret_mse = tf.reduce_mean(tf.losses.huber_loss(secret_true, secret_pred, delta=0.5))
            # cover_mse = tf.reduce_mean(tf.losses.huber_loss(cover_true, cover_pred, delta=0.5))
            secret_mse = tf.reduce_mean(tf.losses.absolute_difference(secret_true, secret_pred))
            cover_mse = tf.reduce_mean(tf.losses.absolute_difference(cover_true, cover_pred))
        with tf.variable_scope("ssim_losses"):
            #secret_ssim = 1. - tf.reduce_mean(tf.image.ssim(secret_true, secret_pred, max_val=1.0))
            #cover_ssim = 1. - tf.reduce_mean(tf.image.ssim(cover_true, cover_pred, max_val=1.0))

            secret_ssim = 1. - (tf.reduce_mean(tf.image.ssim(secret_true[:,:,:,:1],secret_pred[:,:,:,:1], max_val=1.0)) + tf.reduce_mean(tf.image.ssim(secret_true[:,:,:,1:2],secret_pred[:,:,:,1:2], max_val=1.0)) + tf.reduce_mean(tf.image.ssim(secret_true[:,:,:,2:],secret_pred[...,2:], max_val=1.0)))/3.
            cover_ssim = 1. - (tf.reduce_mean(tf.image.ssim(cover_true[:,:,:,:1],cover_pred[:,:,:,:1], max_val=1.0)) + tf.reduce_mean(tf.image.ssim(cover_true[:,:,:,1:2],cover_pred[:,:,:,1:2], max_val=1.0)) + tf.reduce_mean(tf.image.ssim(cover_true[:,:,:,2:],cover_pred[:,:,:,2:], max_val=1.0)))/3.

        # D_final_loss = cover_mse + secret_mse + secret_ssim + cover_ssim + D_loss
        # D_final_loss = D_loss
        G_final_loss = 5*cover_mse + 5*secret_mse + secret_ssim + cover_ssim
        # G_final_loss = cover_mse + secret_mse + secret_ssim + cover_ssim
        # return D_final_loss, G_final_loss, D_loss, G_loss, secret_mse, cover_mse, secret_ssim, cover_ssim
        return G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim

    def get_tensor_to_img_op(self,tensor):
        with tf.variable_scope("",reuse=True):
            # t = tensor*tf.convert_to_tensor([0.229, 0.224, 0.225]) + tf.convert_to_tensor([0.485, 0.456, 0.406])
            tensor = yuv2rgb_tf(tensor)
            return tf.clip_by_value(tensor,0,1)
            # return tf.clip_by_value(tensor,0,255)

    def prepare_training_graph(self,secret_tensor,cover_tensor,global_step_tensor):
        hidden = self.get_hiding_network_op(cover_tensor=cover_tensor, secret_tensor=secret_tensor, is_training=True)
        reveal_output_op = self.get_reveal_network_op(hidden, is_training=True)

        G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim = self.get_loss_op(secret_tensor,reveal_output_op,cover_tensor,hidden)

        global_variables = tf.global_variables()
        gan_varlist = [i for i in global_variables if i.name.startswith('Discriminator')]
        en_de_code_varlist = [i for i in global_variables if i not in gan_varlist]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_op = optimiser.minimize(loss, global_step=global_step)
            # D_minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(D_final_loss, var_list=gan_varlist, global_step=global_step_tensor)
            G_minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(G_final_loss, var_list=en_de_code_varlist, global_step=global_step_tensor)
            # G_minimize_op = tf.train.AdamOptimizer(self.learning_rate).minimize(G_final_loss, global_step=global_step_tensor)

        # tf.summary.scalar('D_loss', D_final_loss,family='train')
        tf.summary.scalar('G_loss', G_final_loss,family='train')
        tf.summary.scalar('secret_mse', secret_mse,family='train')
        tf.summary.scalar('cover_mse', cover_mse,family='train')
        tf.summary.scalar('learning_rate', self.learning_rate,family='train')

        tf.summary.scalar('secret_ssim', secret_ssim)
        tf.summary.scalar('cover_ssim', cover_ssim)

        tf.summary.image('secret',self.get_tensor_to_img_op(secret_tensor),max_outputs=1,family='train')
        tf.summary.image('cover',self.get_tensor_to_img_op(cover_tensor),max_outputs=1,family='train')
        tf.summary.image('hidden',self.get_tensor_to_img_op(hidden),max_outputs=1,family='train')
        # tf.summary.image('hidden_noisy',self.get_tensor_to_img_op(noise_add_op),max_outputs=1,family='train')
        tf.summary.image('revealed',self.get_tensor_to_img_op(reveal_output_op),max_outputs=1,family='train')

        merged_summary_op = tf.summary.merge_all()

        return G_minimize_op, G_final_loss, merged_summary_op, secret_mse,cover_mse, secret_ssim, cover_ssim

    def prepare_test_graph(self,secret_tensor,cover_tensor):
        # y_output, hiding_output_op = self.get_hiding_network_op(cover_tensor=cover_tensor,secret_tensor=secret_tensor, is_training=True)
        hidden = self.get_hiding_network_op(cover_tensor=cover_tensor,secret_tensor=secret_tensor, is_training=False)
        # reveal_output_op = self.get_reveal_network_op(y_output, is_training=True)
        reveal_output_op = self.get_reveal_network_op(hidden, is_training=False)

        G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim = self.get_loss_op(secret_tensor,reveal_output_op,cover_tensor,hidden)
        # tf.summary.scalar('loss', loss_op,family='test')
        # tf.summary.scalar('reveal_net_loss', secret_loss_op,family='test')
        # tf.summary.scalar('cover_net_loss', cover_loss_op,family='test')
        #
        # tf.summary.image('secret',self.get_tensor_to_img_op(secret_tensor),max_outputs=1,family='test')
        # tf.summary.image('cover',self.get_tensor_to_img_op(cover_tensor),max_outputs=1,family='test')
        # tf.summary.image('hidden',self.get_tensor_to_img_op(hiding_output_op),max_outputs=1,family='test')
        # tf.summary.image('revealed',self.get_tensor_to_img_op(reveal_output_op),max_outputs=1,family='test')

        # merged_summary_op = tf.summary.merge_all()

        return hidden, reveal_output_op, G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim

    def save_chkp(self,path):
        global_step = self.sess.run(self.global_step_tensor)
        self.saver.save(self.sess,path,global_step)

    def load_chkp(self,path):
        self.saver.restore(self.sess,path)
        print("LOADED")
        
    def train(self):

        with tf.device('/cpu:0'):
            # segdl = VocRgbDataLoader('/home/jion/moliq/Documents/VOC2012/JPEGImages/', 4, (256, 256), (256, 256), 'voc_train.txt', split='train')
            # segdl_val = VocRgbDataLoader('/home/jion/moliq/Documents/VOC2012/JPEGImages/', 4, (256, 256), (256, 256), 'voc_valid.txt', split='val')
            #segdl = LfwRgbDataLoader('/home/jion/moliq/Documents/lfw/', 2, (256, 256), (256, 256),
            #                         'dataset/lfw_train.txt', split='train')
            #segdl_val = LfwRgbDataLoader('/home/jion/moliq/Documents/lfw/', 2, (256, 256), (256, 256),
            #                             'dataset/lfw_valid.txt', split='val')
            segdl = ImageNetRgbDataLoader('/home/jion/moliq/Documents/imagenet/ILSVRC2012_img_val/', 4, (256, 256), (256, 256),
                                      'dataset/imagenet_train.txt', split='train')
            segdl_val = ImageNetRgbDataLoader('/home/jion/moliq/Documents/imagenet/ILSVRC2012_img_test/', 4, (256, 256), (256, 256),
                                          'dataset/imagenet_valid.txt', split='val')
            iterator = Iterator.from_structure(segdl.data_tr.output_types, segdl.data_tr.output_shapes)
            iterator_val = Iterator.from_structure(segdl_val.data_tr.output_types, segdl_val.data_tr.output_shapes)
            next_batch = iterator.get_next()
            next_batch_val = iterator_val.get_next()
            training_init_op = iterator.make_initializer(segdl.data_tr)
            training_init_op_val = iterator_val.make_initializer(segdl_val.data_tr)

        steps_per_epoch = segdl.data_len / segdl.batch_size
        steps_per_epoch_val = segdl_val.data_len / segdl_val.batch_size

        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step_tensor,
                                                        steps_per_epoch*15, 0.1, staircase=True)
        self.train_op_G, G_final_loss, self.summary_op, self.secret_mse, self.cover_mse, self.secret_ssim, self.cover_ssim = \
            self.prepare_training_graph(self.secret_yuv, self.cover_yuv, self.global_step_tensor)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=30)

        # beta1_power = self.sess.graph.get_tensor_by_name('beta1_power:0')
        # out = self.sess.run(beta1_power)
        # print('beta1_power ', out)

        # exclude_vars = ['beta1_power:0', 'beta2_power:0', 'global_step:0']
        # exclude_vars = ['']
        # restore_variables = [i for i in tf.global_variables() if not i.name in exclude_vars]
        saver = tf.train.Saver()
        loader = tf.train.latest_checkpoint('logs/0509-0030')
        saver.restore(self.sess, loader)
        print('loaded pretrained model')

        #beta1_power = self.sess.graph.get_tensor_by_name('beta1_power:0')
        #out = self.sess.run(beta1_power)
        #print('beta1_power ', out)

        for epoch in range(1, 1+self.epoches):
            print('epoch %d'%epoch)
            self.sess.run(training_init_op)
            for i in range(steps_per_epoch):

                cover_tensor, secret_tensor = self.sess.run(next_batch)
                _, G_loss, secret_mse, cover_mse, secret_ssim, cover_ssim, summary, global_step = \
                    self.sess.run([self.train_op_G, G_final_loss, self.secret_mse, self.cover_mse, self.secret_ssim, self.cover_ssim, self.summary_op, self.global_step_tensor],
                    feed_dict={self.secret_tensor: secret_tensor, self.cover_tensor: cover_tensor})
                self.writer.add_summary(summary, global_step)

                # if i % 5 == 0:
                #     _,  D_loss, summary = \
                #         self.sess.run([self.train_op_D, D_final_loss, self.summary_op],
                #         feed_dict={self.secret_tensor: secret_tensor,self.cover_tensor: cover_tensor})
                #     self.writer.add_summary(summary, global_step)

                if i % 30 == 0:
                    print('Epoch [{}/{}]  Step [{}/{}] G_Loss {:.4f} encoder_ssim {:.4f} encoder_mse {:.4f}'
                          '  decoder_ssim {:.4f}  decoder_mse {:.4f} '.format(
                        epoch, self.epoches, i, steps_per_epoch, G_loss,
                        cover_ssim, cover_mse, secret_ssim, secret_mse ))

            # run validation
            self.sess.run(training_init_op_val)
            # D_loss_val_this_epoch = []
            G_loss_val_this_epoch = []
            secret_ssim_this_epoch = []
            cover_ssim_this_epoch = []
            for i in range(steps_per_epoch_val):
                cover_tensor_val, secret_tensor_val = self.sess.run(next_batch_val)
                G_loss, secret_mse, cover_mse, secret_ssim, cover_ssim = \
                    self.sess.run([G_final_loss, self.secret_mse,self.cover_mse, self.secret_ssim, self.cover_ssim],
                                                                feed_dict={self.secret_tensor: secret_tensor_val,
                                                                self.cover_tensor: cover_tensor_val})
                # D_loss_val_this_epoch.append(D_loss)
                G_loss_val_this_epoch.append(G_loss)
                secret_ssim_this_epoch.append(secret_ssim)
                cover_ssim_this_epoch.append(cover_ssim)
            # mean_D_loss_val_this_epoch = sum(D_loss_val_this_epoch) / len(D_loss_val_this_epoch)
            mean_G_loss_val_this_epoch = sum(G_loss_val_this_epoch) / len(G_loss_val_this_epoch)
            mean_secret_ssim_this_epoch = sum(secret_ssim_this_epoch) / len(secret_ssim_this_epoch)
            mean_cover_ssim_this_epoch = sum(cover_ssim_this_epoch) / len(cover_ssim_this_epoch)
            # print('global step: %d, validation loss: %.4f'%(global_step, mean_loss_val_this_epoch))
            print('VALIDATION Epoch {} global step {}  G_Loss {:.4f} encoder_ssim {:.4f} decoder_ssim {:.4f}'.format(
                epoch, global_step, mean_G_loss_val_this_epoch,
                mean_cover_ssim_this_epoch, mean_secret_ssim_this_epoch))

            # self.save_chkp(self.log_path+'%d_%.3f.ckpt'%(epoch, mean_loss_val_this_epoch))
            self.save_chkp(self.log_path)


    def test_performance(self, log_path):

        hiding_output_op, reveal_output_op, G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim = \
                                self.prepare_test_graph(self.secret_yuv, self.cover_yuv)

        loader = tf.train.latest_checkpoint(log_path)

        # from tensorflow.python.tools import inspect_checkpoint as chkp
        # chkp.print_tensors_in_checkpoint_file(loader, tensor_name='', all_tensors=True)
        # from inspect_checkpoint import print_tensors_in_checkpoint_file
        # print_tensors_in_checkpoint_file(loader, tensor_name='', all_tensors=True)

        # variables = [i for i in tf.global_variables() if i.name not in ['global_step:0']]
        # saver_variables_dict = {value.name[:-2]:value for value in variables}
        # custom_saver = tf.train.Saver(saver_variables_dict)
        # custom_saver.restore(self.sess, loader)
        # print('load model %s'%loader)

        # self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, loader)
        print('load model %s'%loader)

        with tf.device('/cpu:0'):
            # segdl_val = VocRgbDataLoader('/home/jion/moliq/Documents/VOC2012/JPEGImages/', 16, (256, 256), (256, 256), 'voc_valid.txt', split='val')
            segdl_val = LfwRgbDataLoader('/home/jion/moliq/Documents/lfw/', 16, (256, 256), (256, 256),
                                         'dataset/lfw_valid.txt', split='val')
            iterator_val = Iterator.from_structure(segdl_val.data_tr.output_types, segdl_val.data_tr.output_shapes)
            next_batch_val = iterator_val.get_next()
            training_init_op_val = iterator_val.make_initializer(segdl_val.data_tr)

        steps_per_epoch_val = segdl_val.data_len / segdl_val.batch_size

        loss_val_this_epoch = []
        secret_mse_val_this_epoch = []
        cover_mse_val_this_epoch = []
        secret_ssim_this_epoch = []
        cover_ssim_this_epoch = []


        self.sess.run(training_init_op_val)

        # self.saver.restore(self.sess, loader)
        # print('load model %s'%loader)


        for i in range(steps_per_epoch_val):
            cover_tensor_val, secret_tensor_val = self.sess.run(next_batch_val)
            stego, secret_reveal, loss_value, secret_mse_value, cover_mse_value, secret_ssim_value, cover_ssim_value = \
                self.sess.run([hiding_output_op, reveal_output_op, G_final_loss, secret_mse, cover_mse, secret_ssim, cover_ssim],
                              feed_dict={self.secret_tensor: secret_tensor_val,
                                         self.cover_tensor: cover_tensor_val})

            cover_names = segdl_val.imgs_files[i*segdl_val.batch_size:(i+1)*segdl_val.batch_size]
            secret_names = segdl_val.labels_files[i*segdl_val.batch_size:(i+1)*segdl_val.batch_size]

            loss_val_this_epoch.append(loss_value)
            secret_mse_val_this_epoch.append(secret_mse_value)
            cover_mse_val_this_epoch.append(cover_mse_value)
            secret_ssim_this_epoch.append(secret_ssim_value)
            cover_ssim_this_epoch.append(cover_ssim_value)
            if i%10 == 0:
                print('%d %.3f %.3f %.3f %.3f %.3f'%(i, loss_value, secret_mse_value, cover_mse_value, secret_ssim_value, cover_ssim_value))
                save_test_images(cover_names, secret_names, cover_tensor_val, secret_tensor_val, stego, secret_reveal, log_path)
                # np.save('%d %.3f %.3f %.3f %.3f %.3f_cover.npy'%(i, loss_value, secret_mse_value, cover_mse_value, secret_ssim_value, cover_ssim_value), cover_tensor_val)
                # np.save('%d %.3f %.3f %.3f %.3f %.3f_secret.npy'%(i, loss_value, secret_mse_value, cover_mse_value, secret_ssim_value, cover_ssim_value), secret_tensor_val)
                # np.save('%d %.3f %.3f %.3f %.3f %.3f_stego.npy'%(i, loss_value, secret_mse_value, cover_mse_value, secret_ssim_value, cover_ssim_value), stego)
                # np.save('%d %.3f %.3f %.3f %.3f %.3f_secret_reveal.npy'%(i, loss_value, secret_mse_value, cover_mse_value, secret_ssim_value, cover_ssim_value), secret_reveal)


        # mean_loss_val_this_epoch = sum(loss_val_this_epoch) / len(loss_val_this_epoch)
        # mean_secret_mse_val_this_epoch = sum(secret_mse_val_this_epoch) / len(secret_mse_val_this_epoch)
        # mean_cover_mse_val_this_epoch = sum(cover_mse_val_this_epoch) / len(cover_mse_val_this_epoch)
        # mean_secret_ssim_this_epoch = sum(secret_ssim_this_epoch) / len(secret_ssim_this_epoch)
        # mean_cover_ssim_this_epoch = sum(cover_ssim_this_epoch) / len(cover_ssim_this_epoch)

        mean_loss_val_this_epoch = np.mean(loss_val_this_epoch)
        mean_secret_mse_val_this_epoch = np.mean(secret_mse_val_this_epoch)
        mean_cover_mse_val_this_epoch = np.mean(cover_mse_val_this_epoch)
        mean_secret_ssim_this_epoch = np.mean(secret_ssim_this_epoch)
        mean_cover_ssim_this_epoch = np.mean(cover_ssim_this_epoch)

        print('validation loss: %.4f' % mean_loss_val_this_epoch)
        print('secret mse: %.4f' % mean_secret_mse_val_this_epoch)
        print('cover mse : %.4f' % mean_cover_mse_val_this_epoch)
        print('secret ssim: %.4f' % mean_secret_ssim_this_epoch)
        print('cover ssim: %.4f' % mean_cover_ssim_this_epoch)


if __name__ == '__main__':
    train_model = Model()
    train_model.train()
    # train_model.test_performance(train_model.log_path)
    # train_model.test_performance('logs/0427-1506')
    # train_model.test_performance('logs/0428-2048')
    # train_model.test_performance('logs/0505-1617')

                

