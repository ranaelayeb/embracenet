import argparse
import copy
import os

import numpy as np
import tensorflow as tf

import context
from embracenet_tf2 import EmbraceNet


class BimodalMNISTModel():

  def parse_args(self, args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--model_dropout', action='store_true', help='Specify this to employ modality dropout during training.')
    parser.add_argument('--model_drop_left', action='store_true', help='Specity this to drop left-side modality.')
    parser.add_argument('--model_drop_right', action='store_true', help='Specity this to drop right-side modality.')

    self.args, remaining_args = parser.parse_known_args(args=args)
    return copy.deepcopy(self.args), remaining_args
  

  def prepare(self, is_training, global_step=0):
    # config. parameters
    self.global_step = global_step

    # main model
    self.model = EmbraceNetBimodalModel(is_training=is_training, args=self.args, name='embracenet_bimodal')
    if (is_training):
      self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=self.args.model_learning_rate
      )
      self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # checkpoint
    if (is_training):
      self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, step=tf.Variable(1))
    else:
      self.ckpt = tf.train.Checkpoint(model=self.model)

      
  def save(self, base_path):
    save_path = os.path.join(base_path, 'ckpt_%d' % (self.global_step))
    self.model.save_weights(save_path)


  def restore(self, ckpt_path):
    self.model.load_weights(ckpt_path)
  

  def get_model(self):
    return self.model


  def train_step(self, input_list, truth_list, summary=None):
    # input processing
    input_tensor = tf.convert_to_tensor(np.array(input_list), dtype=tf.dtypes.float32)
    truth_tensor = tf.convert_to_tensor(np.array(truth_list), dtype=tf.dtypes.int64)

    # do forward propagation
    with tf.GradientTape() as tape:
      output_tensor = self.model(input_tensor)
      loss = self.loss_fn(truth_tensor, output_tensor)
    
    # do back propagation
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    # finalize
    self.global_step += 1
    
    # write summary
    if (summary is not None):
      with summary.as_default():
        tf.summary.scalar('loss', loss, step=self.global_step)
    
    return loss


  def predict(self, input_list):
    # get output
    output_list = self.model(input_list)
    prob_list = tf.nn.softmax(output_list)
    class_list = tf.math.argmax(prob_list, axis=-1)

    # finalize
    return prob_list, class_list



class EmbraceNetBimodalModel(tf.keras.Model):
  def __init__(self, is_training, args, **kwargs):
    super(EmbraceNetBimodalModel, self).__init__(**kwargs)
    
    # input parameters
    self.is_training = is_training
    self.args = args

    # pre embracement layers
    self.pre_left = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
      tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
      tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
    ])
    self.pre_right = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
      tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
      tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
    ])
    self.pre_output_size = (28 * 14 * 64) // 4

    # embracenet
    self.embracenet = EmbraceNet(input_size_list=[self.pre_output_size, self.pre_output_size], embracement_size=512, name='embracenet')

    # post embracement layers
    self.post = tf.keras.layers.Dense(units=10)

  
  def call(self, x):
    # separate x into left/right
    x_left = tf.expand_dims(x[:, 0], axis=-1)
    x_left = self.pre_left(x_left)
    x_left = tf.reshape(x_left, [-1, self.pre_output_size])

    x_right = tf.expand_dims(x[:, 1], axis=-1)
    x_right = self.pre_right(x_right)
    x_right = tf.reshape(x_right, [-1, self.pre_output_size])

    # drop left or right modality
    availabilities = None
    if (self.args.model_drop_left or self.args.model_drop_right):
      availabilities = tf.ones([x.shape[0], 2])
      if (self.args.model_drop_left):
        availabilities[:, 0] = 0
      if (self.args.model_drop_right):
        availabilities[:, 1] = 0

    # dropout during training
    if (self.is_training and self.args.model_dropout):
      dropout_prob = tf.random.uniform([])
      
      def _dropout_modalities():
        target_modalities = tf.cast(tf.math.round(tf.random.uniform([x.shape[0]]) * 2.0), tf.dtypes.int64)
        return tf.one_hot(target_modalities, depth=2, dtype=tf.dtypes.float32)
      
      availabilities = tf.cond(tf.less(dropout_prob, 0.5), _dropout_modalities, lambda: tf.ones([x.shape[0], 2]))

    # embrace
    x_embrace = self.embracenet([x_left, x_right], availabilities=availabilities)

    # employ final layers
    x = self.post(x_embrace)

    # finalize
    return x
