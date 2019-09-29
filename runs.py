# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:32:07 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

resnet runs
"""
import tensorflow as tf
from cifar10 import Cifar10
from collections import OrderedDict
from networks import cycle_lr
import argparse
import os
import numpy as np
from models import VGG16
np.random.seed(0)
tf.set_random_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    # optim config
    parser.add_argument('--model_name', type=str, default = 'ResNet')
    parser.add_argument('--datasets', type = str, default = 'CIFAR10')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type = float, default = 0.01)
    parser.add_argument('--max_lr', type = float, default = 0.2)
    parser.add_argument('--cycle_epoch', type = int, default = 20)
    parser.add_argument('--cycle_ratio', type = float, default = 0.7)
    parser.add_argument('--num_classes', type = int, default = 10)

    args = parser.parse_args()
    
    config = OrderedDict([
        ('model_name', args.model_name),
        ('datasets', args.datasets),
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('max_lr', args.max_lr),
        ('cycle_epoch', args.cycle_epoch),
        ('cycle_ratio', args.cycle_ratio),
        ('num_classes', args.num_classes)])

    return config


config = parse_args()


### call data ###
cifar10 = Cifar10()
n_samples = cifar10.num_examples 
n_test_samples = cifar10.num_test_examples

### call models ###
model = VGG16(config['num_classes'])   


### make folder ###
mother_folder = config['model_name']
try:
    os.mkdir(mother_folder)
except OSError:
    pass    


### outputs ###
pred, loss = model.Forward()

iter_per_epoch = int(n_samples/config['batch_size']) 


### cyclic learning rate ###
Lr = cycle_lr(config['base_lr'], config['max_lr'], iter_per_epoch, 
              config['cycle_epoch'], config['cycle_ratio'], config['epochs'])

cy_lr = tf.placeholder(tf.float32, shape=(),  name = "cy_lr")


### run ###
folder_name = os.path.join(mother_folder, config['model_name']+'_'+config['datasets'])


with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cy_lr).minimize(loss)

prediction = tf.argmax(pred, 1)
correct_prediction = tf.equal(prediction, tf.argmax(model.y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
tf.summary.scalar('accuracy', accuracy)


with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    
    try:
        os.mkdir(folder_name)
    except OSError:
        pass    
    
    iteration = 0
    iter_per_test_epoch = n_test_samples/config['batch_size'] 
    for epoch in range(config['epochs']):
        epoch_loss = 0.
        epoch_acc = 0.
        for iter_in_epoch in range(iter_per_epoch):
            epoch_x, epoch_y = cifar10.next_train_batch(config['batch_size'])
            _, c, acc = sess.run([optimizer, loss, accuracy], 
                            feed_dict = {model.x: epoch_x, model.y: epoch_y, 
                                         model.training:True, model.keep_prob:0.7,
                                         cy_lr: Lr[iteration]})
            epoch_loss += c
            epoch_acc += acc
            iteration+=1
            if iter_in_epoch%100 == 0:
                print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
                      'completed out of ', config['epochs'], 'loss: ', epoch_loss/(iter_in_epoch+1),
                      'acc: ', '{:.2f}%'.format(epoch_acc*100/(iter_in_epoch+1)))
        print('######################')        
        print('TRAIN')        
        print('Epoch ', epoch, '{:.2f}%'.format(100*(iter_in_epoch+1)/int(iter_per_epoch)),
              'completed out of ', config['epochs'], 'loss: ', epoch_loss/int(iter_per_epoch),
              'acc: ', '{:.2f}%'.format(epoch_acc*100/int(iter_per_epoch)))
        
        test_loss = 0.
        test_acc = 0.
        for iter_in_epoch in range(int(iter_per_test_epoch)):            
            epoch_x, epoch_y = cifar10.next_test_batch(config['batch_size'])  
            c, acc = sess.run([loss, accuracy], 
                              feed_dict = {model.x: epoch_x, model.y: epoch_y, 
                                           model.training:False, model.keep_prob:1.0,
                                           cy_lr: Lr[iteration]})
            test_loss += c
            test_acc += acc
        print('TEST')        
        print('Epoch ', epoch,  'loss: ', test_loss/int(iter_per_test_epoch), 
              'acc: ', '{:.2f}%'.format(test_acc*100/int(iter_per_test_epoch)))
        print('###################### \n')     
        


   

