# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:12:40 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

VGG16 model
"""
import tensorflow as tf
from networks import Batch_norm, Augmentation

class VGG16(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name = "vgg_input")
        self.y = tf.placeholder(tf.float32, shape=(None, self.num_classes), name = "vgg_one_hot")  # onehot
        self.training = tf.placeholder(tf.bool,shape=(), name = "training")
        self.keep_prob = tf.placeholder(tf.float32, name = "drop_out")
    
    def Forward(self):
        self.aug = Augmentation(self.x, self.training)
        
        
        ### conv1 ###
        self.conv1_1 = tf.layers.conv2d(
          inputs=self.aug, filters=64, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv1_1')
        self.relu1_1 = tf.nn.relu(self.conv1_1)
        self.batch1_1 = Batch_norm(self.relu1_1, self.training)
        self.droput1_1 = tf.nn.dropout(self.batch1_1, keep_prob= self.keep_prob)
        
        self.conv1_2 = tf.layers.conv2d(
          inputs=self.droput1_1, filters=64, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv1_2')
        self.relu1_2 = tf.nn.relu(self.conv1_2)
        self.batch1_2 = Batch_norm(self.relu1_2, self.training)
        
        self.pool1 = tf.layers.max_pooling2d(self.batch1_2, pool_size=2,
          strides=2, padding = 'SAME', name = 'pool1')


        ### conv2 ###
        self.conv2_1 = tf.layers.conv2d(
          inputs=self.pool1, filters=128, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv2_1')
        self.relu2_1 = tf.nn.relu(self.conv2_1)
        self.batch2_1 = Batch_norm(self.relu2_1, self.training)
        self.droput2_1 = tf.nn.dropout(self.batch2_1, keep_prob= self.keep_prob)

        self.conv2_2 = tf.layers.conv2d(
          inputs=self.droput2_1, filters=128, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv2_2')
        self.relu2_2 = tf.nn.relu(self.conv2_2)
        self.batch2_2 = Batch_norm(self.relu2_2, self.training)

        self.pool2 = tf.layers.max_pooling2d(self.batch2_2, pool_size=2,
          strides=2, padding = 'SAME', name = 'pool2')
        
        
        ### conv3 ###
        self.conv3_1 = tf.layers.conv2d(
          inputs=self.pool2, filters=256, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv3_1')
        self.relu3_1 = tf.nn.relu(self.conv3_1)
        self.batch3_1 = Batch_norm(self.relu3_1, self.training)
        self.droput3_1 = tf.nn.dropout(self.batch3_1, keep_prob= self.keep_prob)

        self.conv3_2 = tf.layers.conv2d(
          inputs=self.droput3_1, filters=256, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv3_2')
        self.relu3_2 = tf.nn.relu(self.conv3_2)
        self.batch3_2 = Batch_norm(self.relu3_2, self.training)
        self.droput3_2 = tf.nn.dropout(self.batch3_2, keep_prob= self.keep_prob)
        
        self.conv3_3 = tf.layers.conv2d(
          inputs=self.droput3_2, filters=256, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv3_3')
        self.relu3_3 = tf.nn.relu(self.conv3_3)
        self.batch3_3 = Batch_norm(self.relu3_3, self.training)

        self.pool3 = tf.layers.max_pooling2d(self.batch3_3, pool_size=2,
          strides=2, padding = 'SAME', name = 'pool3')
        
        
        ### conv4 ###
        self.conv4_1 = tf.layers.conv2d(
          inputs=self.pool3, filters=512, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv4_1')
        self.relu4_1 = tf.nn.relu(self.conv4_1)
        self.batch4_1 = Batch_norm(self.relu4_1, self.training)
        self.droput4_1 = tf.nn.dropout(self.batch4_1, keep_prob= self.keep_prob)

        self.conv4_2 = tf.layers.conv2d(
          inputs=self.droput4_1, filters=512, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv4_2')
        self.relu4_2 = tf.nn.relu(self.conv4_2)
        self.batch4_2 = Batch_norm(self.relu4_2, self.training)
        self.droput4_2 = tf.nn.dropout(self.batch4_2, keep_prob= self.keep_prob)

        self.conv4_3 = tf.layers.conv2d(
          inputs=self.droput4_2, filters=512, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv4_3')
        self.relu4_3 = tf.nn.relu(self.conv4_3)
        self.batch4_3 = Batch_norm(self.relu4_3, self.training)
 
        self.pool4 = tf.layers.max_pooling2d(self.batch4_3, pool_size=2,
          strides=2, padding = 'SAME', name = 'pool4')
        
        
        ### conv5 ###
        self.conv5_1 = tf.layers.conv2d(
          inputs=self.pool4, filters=512, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv5_1')
        self.relu5_1 = tf.nn.relu(self.conv5_1)
        self.batch5_1 = Batch_norm(self.relu5_1, self.training)
        self.droput5_1 = tf.nn.dropout(self.batch5_1, keep_prob= self.keep_prob)

        self.conv5_2 = tf.layers.conv2d(
          inputs=self.droput5_1, filters=512, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv5_2')
        self.relu5_2 = tf.nn.relu(self.conv5_2)
        self.batch5_2 = Batch_norm(self.relu5_2, self.training)
        self.droput5_2 = tf.nn.dropout(self.batch5_2, keep_prob= self.keep_prob)

        self.conv5_3 = tf.layers.conv2d(
          inputs=self.droput5_2, filters=512, kernel_size=3,
          strides=1, padding = 'SAME', name = 'conv5_3')
        self.relu5_3 = tf.nn.relu(self.conv5_3)
        self.batch5_3 = Batch_norm(self.relu5_3, self.training)
        
        self.pool5 = tf.layers.max_pooling2d(self.batch5_3, pool_size=2,
          strides=2, padding = 'SAME', name = 'pool5')
        
        self.droputf_1 = tf.nn.dropout(self.pool5, keep_prob= self.keep_prob)
        self.reshape = tf.reshape(self.droputf_1, [-1, 1*1*512])

        self.dense1 = tf.layers.dense(inputs=self.reshape, units=512,
                                    name = 'dense1')
        self.reluf_1 = tf.nn.relu(self.dense1)
        self.batchf_1 = Batch_norm(self.reluf_1, self.training)
        
        self.droputf_2 = tf.nn.dropout(self.batchf_1, keep_prob= self.keep_prob)

        self.pred = tf.layers.dense(inputs=self.droputf_2, units=self.num_classes,
                                        name=  'prediction')       
   
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred,
                                                             labels=tf.stop_gradient([self.y])))

        def Summaray():
            tf.summary.scalar('loss', self.loss)

        Summaray()

    
        return self.pred, self.loss
       
        

        

