import tensorflow as tf
from tensorflow.python.client import device_lib
import os



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print(tf.config.list_physical_devices('GPU'))

device = device_lib.list_local_devices()
print('GPU Info:', device[1].physical_device_desc)