import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from utils import yuv2rgb_np
# import tensorflow as tf
# from tensorflow.data import Iterator
# from Dataset import SegDataLoader

def visulize_npy_result(log_dir):
    save_dir = os.path.join(log_dir, 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    secret_revealed = np.load(os.path.join(log_dir, 'test_secrets_revealed.npy'))
    secret_revealed = np.reshape(secret_revealed, (secret_revealed.shape[0]*secret_revealed.shape[1], 256, 256, 1))
    # secret_revealed = (((secret_revealed - secret_revealed.min()) * 255) / (secret_revealed.max() - secret_revealed.min())).astype(np.uint8)
    stegos = np.load(os.path.join(log_dir, 'test_stegos.npy'))
    stegos = np.reshape(stegos, (stegos.shape[0]*stegos.shape[1], 256, 256, 3))
    # stegos = (((stegos - stegos.min()) * 255) / (stegos.max() - stegos.min())).astype(np.uint8)


    with tf.device('/cpu:0'):
        segdl_val = SegDataLoader('/home/moliq/Documents/VOC2012/JPEGImages/', 1, (256, 256), (256, 256),
                                  'voc_valid.txt', split='val')
        iterator_val = Iterator.from_structure(segdl_val.data_tr.output_types, segdl_val.data_tr.output_shapes)
        next_batch_val = iterator_val.get_next()
        training_init_op_val = iterator_val.make_initializer(segdl_val.data_tr)

    with tf.Session() as sess:
        sess.run(training_init_op_val)
        for i in range(segdl_val.data_len):
            print('%d / %d'%(i, segdl_val.data_len))
            cover_name = segdl_val.imgs_files[i]
            secret_name = segdl_val.labels_files[i]
            common_name = os.path.basename(cover_name)[:-4] + '+' + os.path.basename(secret_name)[:-4]
            cover_image, secret_image = sess.run(next_batch_val)
            stego_i = stegos[i]
            stego_i = (((stego_i - stego_i.min()) * 255) / (stego_i.max() - stego_i.min())).astype(np.uint8)
            secret_revealed_i = secret_revealed[i]
            secret_revealed_i = (((secret_revealed_i - secret_revealed_i.min()) * 255) / (secret_revealed_i.max() - secret_revealed_i.min())).astype(np.uint8)

            cv2.imwrite(os.path.join(save_dir, common_name+'_cover.jpg'), np.squeeze(cover_image*255).astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(save_dir, common_name+'_secret.jpg'), np.squeeze(secret_image*255).astype(np.uint8))
            cv2.imwrite(os.path.join(save_dir, common_name+'_stego.jpg'), np.squeeze(stego_i)[:,:,::-1])
            cv2.imwrite(os.path.join(save_dir, common_name+'_secret_reveal.jpg'), np.squeeze(secret_revealed_i))

    print('Done!')


def save_test_images(cover_names, secret_names, covers, secrets, stegos, secret_reveals, log_dir):
    save_dir = os.path.join(log_dir, 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = covers.shape[0]
    stegos = yuv2rgb_np(stegos)
    secret_reveals = yuv2rgb_np(secret_reveals)
    for i in range(batch_size):
        cover = covers[i]
        secret = secrets[i]
        stego = stegos[i]
        secret_reveal = secret_reveals[i]
        cover_name = cover_names[i]
        secret_name = secret_names[i]
        common_name = os.path.basename(cover_name)[:-4] + '+' + os.path.basename(secret_name)[:-4]

        stego = (((stego - stego.min()) * 255) / (stego.max() - stego.min())).astype(np.uint8)
        secret_reveal = (((secret_reveal - secret_reveal.min()) * 255) / (
                secret_reveal.max() - secret_reveal.min())).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, common_name + '_cover.jpg'),
                    np.squeeze(cover * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(os.path.join(save_dir, common_name + '_secret.jpg'),
                    np.squeeze(secret * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(os.path.join(save_dir, common_name + '_stego.jpg'), np.squeeze(stego)[:, :, ::-1])
        cv2.imwrite(os.path.join(save_dir, common_name + '_secret_reveal.jpg'), np.squeeze(secret_reveal)[:, :, ::-1])


def plot_validation_scores(log_path):

    log = open(log_path, 'r').readlines()
    scores = [i.strip().split(' ') for i in log if i.startswith('VALIDATION Epoch')]

    # d_loss = [float(i[-7]) for i in scores]
    # g_loss = [float(i[-5]) for i in scores]
    en_sim = [float(i[-3]) for i in scores]
    de_sim = [float(i[-1]) for i in scores]

    plt.figure()
    n = len(en_sim)
    x = np.linspace(1, n, n)

    l1, = plt.plot(x, en_sim, label='line', color='red', linewidth=1.0)
    l3, = plt.plot(x, de_sim, label='line', color='blue', linewidth=1.0)

    plt.xlabel('epoch')
    plt.ylabel('simm')
    # plt.title('Google-{:s}'.format(name))
    plt.legend(handles=[l1, l3], labels=['cover', 'secret'],
               loc='best')
    plt.savefig(os.path.join(os.path.dirname(log_path), '{:s}_result.jpg'.format('ssim')))

    # plt.figure()
    # n = len(en_sim)
    # x = np.linspace(1, n, n)
    # l2, = plt.plot(x, d_loss, label='parabola', color='red', linewidth=1.0, linestyle='--')
    # l4, = plt.plot(x, g_loss, label='parabola', color='blue', linewidth=1.0, linestyle='--')
    #
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    #
    # plt.legend(handles=[l2, l4], labels=['d_loss', 'g_loss'],
    #            loc='best')
    # plt.savefig(os.path.join(os.path.dirname(log_path), '{:s}_result.jpg'.format('loss')))


if __name__ == '__main__':
    # visulize_npy_result('logs/0403-0019')
    # visulize_npy_result('logs/0407-2119')
    plot_validation_scores('logs/0509-0030/log.txt')