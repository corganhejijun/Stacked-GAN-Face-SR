# -*- coding: utf-8 -*- 
import tensorflow as tf
from glob import glob
import os
import time
import numpy as np
import random

from .ops import *
from .utils import *

class ScaleGan(object):
    def __init__(self, sess, dataset_name, origin_size=64, img_size=256):
        self.dataset_name = dataset_name
        self.batch_size = 50
        self.checkpoint_dir = './checkpoint'
        self.img_dim = 3 # image file color channel
        self.conv_dim = 64
        self.sess = sess
        self.img_size = img_size 
        self.L1_lambda = 100
        self.LAMBDA = 10 # Gradient penalty lambda hyperparameter
        self.origin_size = origin_size
        self.build_model()

    def build_model(self):
        self.input_img = tf.placeholder(tf.float32, 
            [self.batch_size, self.img_size, self.img_size, self.img_dim*2],
            name='input_A_and_B_images')
        sizes = []
        size = self.origin_size
        while(size < self.img_size):
            sizes.append(size)
            size = size*2
        if size != self.img_size:
            print("size error when building model. origin size is " 
                    + str(self.origin_size) + " and target image size is " + str(self.img_size))
            return
        # A is sample, B is ground truth, 
        self.real_A = []
        A = self.input_img[:, :, :, :self.img_dim]
        for size in sizes:
            self.real_A.append(tf.image.resize_images(A, (size, size)))
        self.real_A.append(A)
        self.real_B = []
        B = self.input_img[:, :, :, self.img_dim:2*self.img_dim]
        for size in sizes:
            self.real_B.append(tf.image.resize_images(B, (size, size)))
        self.real_B.append(B)

        self.fake_B = self.generator(self.real_A[-1])
        self.fake_sample = self.sampler(self.real_A[-1])

        self.real_AB = []
        for i in range(len(self.real_A)):
            self.real_AB.append(tf.concat([self.real_A[i], self.real_B[i]], 3))
        self.fake_AB = []
        for i in range(len(self.real_A)):
            self.fake_AB.append(tf.concat([self.real_A[i], self.fake_B[i]], 3))
        self.both_BB = []
        for i in range(len(self.real_B)):
            self.both_BB.append(tf.concat([self.real_B[i], self.fake_B[i]], 3))

        self.D_real_AB, self.D_real_AB_logits = self.discriminator(self.real_AB, name="discriminator")
        self.D_fake_AB, self.D_fake_AB_logits = self.discriminator(self.fake_AB, name="discriminator", reuse=True)
        self.D_both_BB, self.D_both_BB_logits = self.discriminator(self.both_BB, name="discriminator", reuse=True)

        self.D_sum_real_AB = []
        for i in range(len(self.D_real_AB)):
            self.D_sum_real_AB.append(tf.summary.histogram("d_real_AB_" + str(i), self.D_real_AB[i]))
        self.D_sum_fake_AB = []
        for i in range(len(self.D_fake_AB)):
            self.D_sum_fake_AB.append(tf.summary.histogram("d_fake_AB_" + str(i), self.D_fake_AB[i]))
        self.D_sum_both_BB = []
        for i in range(len(self.D_both_BB)):
            self.D_sum_both_BB.append(tf.summary.histogram("d_both_BB_" + str(i), self.D_both_BB[i]))
        self.fake_B_sum = []
        for i in range(len(self.fake_B)):
            self.fake_B_sum.append(tf.summary.image("fake_B_" + str(i), self.fake_B[i]))

        self.d_loss_real = []
        for i in range(len(self.D_real_AB)):
            self.d_loss_real.append(tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_AB_logits[i], labels=tf.ones_like(self.D_real_AB[i]))))
        self.d_loss_fake = []
        for i in range(len(self.D_fake_AB)):
            self.d_loss_fake.append(tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_AB_logits[i], labels=tf.zeros_like(self.D_fake_AB[i]))))
        self.d_loss_both = []
        for i in range(len(self.D_both_BB)):
            self.d_loss_both.append(tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_both_BB_logits[i], labels=tf.zeros_like(self.D_both_BB[i]))))
        self.g_loss = []
        for i in range(len(self.D_both_BB)):
            if i == 0:
                self.g_loss.append(
                    tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_both_BB_logits[i], labels=tf.ones_like(self.D_both_BB[i])))
                    + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B[i] - self.fake_B[i]))
                    + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_AB_logits[i], labels=tf.ones_like(self.D_fake_AB[i])))
                )
            else:
                self.g_loss.append(
                    self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B[i] - self.fake_B[i]))
                    + tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_AB_logits[i], labels=tf.ones_like(self.D_fake_AB[i])))
                )
        
        self.d_loss_real_sum = []
        for i in range(len(self.d_loss_real)):
            self.d_loss_real_sum.append(tf.summary.scalar("d_loss_real_" + str(i), self.d_loss_real[i]))
        self.d_loss_fake_sum = []
        for i in range(len(self.d_loss_fake)):
            self.d_loss_fake_sum.append(tf.summary.scalar("d_loss_fake_" + str(i), self.d_loss_fake[i]))
        self.d_loss_both_sum = []
        for i in range(len(self.d_loss_both)):
            self.d_loss_both_sum.append(tf.summary.scalar("d_loss_both_" + str(i), self.d_loss_both[i]))

        self.d_loss = []
        for i in range(len(self.d_loss_real)):
            if i == 0:
                self.d_loss.append(self.d_loss_real[i] + self.d_loss_fake[i] + self.d_loss_both[i])
            else:
                self.d_loss.append(self.d_loss_real[i] + self.d_loss_fake[i])
        self.d_loss_sum = []
        for i in range(len(self.d_loss)):
            self.d_loss_sum.append(tf.summary.scalar("d_loss_" + str(i), self.d_loss[i]))
        self.g_loss_sum = []
        for i in range(len(self.d_loss)):
            self.g_loss_sum.append(tf.summary.scalar("g_loss_" + str(i), self.g_loss[i]))
            
        t_vars = tf.trainable_variables()
        self.d_vars = []
        for var in t_vars:
            if 'd_' in var.name:
                self.d_vars.append(var)
        self.g_vars = []
        for var in t_vars:
            if 'g_' in var.name:
                self.g_vars.append(var)

        self.saver = tf.train.Saver()       
        
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.img_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
            
    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.img_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*.jpg'.format(self.dataset_name)), self.batch_size)
        sample = [load_data(sample_file, self.img_size, self.img_size + int(self.img_size/8)) for sample_file in data]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images
        
    def sample_model(self, sample_dir, epoch):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_sample, self.d_loss[-1], self.g_loss[-1]], feed_dict={self.input_img: sample_images}
        )
        save_merge_images(samples, [self.batch_size, 1], './{}/train_{:02d}.png'.format(sample_dir, epoch))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
    
    def train(self, args):
        d_optim = []
        for i in range(len(self.d_loss)):
            d_optim.append(tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_loss[i], var_list=self.d_vars))
        g_optim = []
        for i in range(len(self.g_loss)):
            g_optim.append(tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_loss[i], var_list=self.g_vars))

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        g_sum_list = []
        for i in range(len(self.D_sum_fake_AB)):
            g_sum_list.append(self.D_sum_fake_AB[i])
            g_sum_list.append(self.D_sum_both_BB[i])
            g_sum_list.append(self.fake_B_sum[i])
            g_sum_list.append(self.d_loss_fake_sum[i])
            g_sum_list.append(self.d_loss_both_sum[i])
            g_sum_list.append(self.g_loss_sum[i])
        self.g_sum = tf.summary.merge(g_sum_list)
        d_sum_list = []
        for i in range(len(self.D_sum_real_AB)):
            d_sum_list.append(self.D_sum_real_AB[i])
            d_sum_list.append(self.d_loss_real_sum[i])
            d_sum_list.append(self.d_loss_sum[i])
        self.d_sum = tf.summary.merge(d_sum_list)

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter = 1
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            
        for epoch in range(args.epoch):
            data = glob('./datasets/{}/train/*.jpg'.format(self.dataset_name))
            random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size
            for idx in range(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_data(batch_file, self.img_size, self.img_size+int(self.img_size/8)) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                for i in range(len(d_optim)):
                    _, summary_str = self.sess.run([d_optim[i], self.d_sum], feed_dict={self.input_img: batch_images})
                    self.writer.add_summary(summary_str, counter)
                    _, summary_str = self.sess.run([g_optim[i], self.g_sum], feed_dict={self.input_img: batch_images})
                    self.writer.add_summary(summary_str, counter)
                    _, summary_str = self.sess.run([g_optim[i], self.g_sum], feed_dict={self.input_img: batch_images})
                    self.writer.add_summary(summary_str, counter)
                errD = ""
                errG = ""
                for i in range(len(self.d_loss)):
                    result = self.d_loss[i].eval({self.input_img: batch_images})
                    errD += "{:.4f},".format(result)
                    result = self.g_loss[i].eval({self.input_img: batch_images})
                    errG += "{:.4f},".format(result)
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: [%s], g_loss: [%s]" \
                        % (epoch, idx, batch_idxs, time.time() - start_time, errD[:-1], errG[:-1]))
                if np.mod(counter, 500) == 2:
                    self.save(args.checkpoint_dir, counter)
                        
            self.sample_model(args.sample_dir, epoch)


    def discriminator(self, img, name, reuse=False):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            D = []
            D_logits = []
            for i in range(len(img)):
                h = img[i]
                size = h.shape[1].value
                count = 0
                while size > 8:
                    fact = 2**count
                    if fact > 8:
                        fact = 8
                    h = batch_norm(
                        conv2d(h, self.conv_dim*fact, name="d_" + str(i) + "_h" + str(count) + "_conv"),
                        name="d_" + str(i) + "_bn" + str(count))
                    h = lrelu(h)
                    size = h.shape[1].value
                    count += 1
                D.append(tf.nn.sigmoid(h))
                D_logits.append(h)
            return D, D_logits

    def scaleImage(self, img):
        e = conv2d(img, self.conv_dim, name='g_e0_conv')
        size = e.shape[1].value
        count = 1
        eList = [e]
        # from img_size to 1x1
        while size > 1:
            fact = 2**count
            if fact > 8:
                fact = 8
            e = batch_norm(conv2d(lrelu(e), self.conv_dim*fact, name="g_e" + str(count) + "_conv"), name="g_bn_e" + str(count) + "_conv")
            size = e.shape[1].value
            count += 1
            if size == 1:
                break
            eList.append(e)
        # e is 1x1
        d = e
        d_for_out = None
        # from 1x1 to origin_size
        while size < self.origin_size:
            size = size*2
            d_for_out = d
            fact = 2**count
            if fact > 8:
                fact = 8
            d = deconv2d(tf.nn.relu(d), [self.batch_size, size, size, self.conv_dim*fact],
                            name="g_d" + str(count) + "_deconv")
            d = batch_norm(d, name="g_bn_d" + str(count) + "_deconv")
            d = tf.concat([d, eList[-1]], 3)
            del eList[-1]
            count -= 1
        # from origin_size to img_size
        fakeB = []
        while size <= self.img_size:
            B = deconv2d(tf.nn.relu(d_for_out), [self.batch_size, size, size, self.img_dim],
                            name="g_d" + str(count) + "_out")
            fakeB.append(tf.nn.tanh(B))
            size = size*2
            count -=1
            rb = multi_residual_block(d, size)
            d_for_out = rb
            fact = 2**count
            if fact > 8:
                fact = 8
            if size < self.img_size:
                d = deconv2d(tf.nn.relu(rb), [self.batch_size, size, size, self.conv_dim*fact],
                                name="g_d" + str(count) + "_deconv")
                d = batch_norm(d, name="g_bn_d" + str(count) + "_deconv")
                d = tf.concat([d, eList[-1]], 3)
                del eList[-1]
        return fakeB

    def generator(self, img):
        with tf.variable_scope("generator"):
            return self.scaleImage(img)

    def sampler(self, img):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            fakeB = self.scaleImage(img)
            return fakeB[-1]

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        sample_files_all = glob('./datasets/{}/val_test/*.jpg'.format(self.dataset_name))
        
        max_size = 10000
        batch_count = 0
        while len(sample_files_all) > max_size * batch_count:
            endIdx = max_size * (batch_count + 1)
            if (len(sample_files_all) < endIdx):
                endIdx = len(sample_files_all)
            sample_files = sample_files_all[batch_count * max_size : endIdx]
            print("Loading testing images ... from {0} to {1} of total {2}".format(batch_count * max_size, endIdx, len(sample_files_all)))
            batch_count += 1
            sample = [load_data(sample_file, self.img_size, self.img_size+int(self.img_size/8), is_test=True) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            sample_images = [sample_images[i:i+self.batch_size] for i in range(0, len(sample_images), self.batch_size)]
            sample_images = np.array(sample_images)

            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            print("file number: {}".format(len(sample_files)))

            for i, sample_image in enumerate(sample_images):
                idx = i
                fileName = sample_files[idx].split('/')[-1].split('.jpg')[0]
                print("sampling image {}, {} of total {}".format(fileName, idx + (batch_count - 1) * max_size, len(sample_files_all) // self.batch_size))
                samples = self.sess.run(self.fake_sample, feed_dict={self.input_img: sample_image})
                for j in range(self.batch_size):
                    jdx = j+i*self.batch_size
                    save_images(samples[j, :, :, :], './{}/{}.png'.format(args.test_dir, sample_files[jdx].split('/')[-1].split('.jpg')[0]))