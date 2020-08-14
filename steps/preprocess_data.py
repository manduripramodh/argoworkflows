import pickle
import tensorflow as tf
import os

root_path='/training'

if __name__ == '__main__':
  path = '{}/input_data/mnist.npz'.format(root_path)
  print(path)
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path)

  os.makedirs('{}/output/'.format(root_path), exist_ok=True)
  with open('{}/output/train.pickle'.format(root_path), 'wb') as f:
      pickle.dump([train_images, train_labels], f)

  with open('{}/output/test.pickle'.format(root_path), 'wb') as f:
      pickle.dump([test_images, test_labels], f)