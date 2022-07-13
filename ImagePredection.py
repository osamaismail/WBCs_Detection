import numpy as np
import os
import pathlib
import tensorflow as tf
from tensorflow import keras


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
model = keras.models.load_model('WBCs5.h5')
pathTrain = 'TRAIN'
path = 'tests'
myList = os.listdir(path)
img_height = 180
img_width = 180
data_dir = pathlib.Path(pathTrain)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=32)

class_names = train_ds.class_names

model.summary()

for i in myList:
    cellPath = path+'/'+i
    img = keras.preprocessing.image.load_img(
        cellPath, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        f"This image {i} most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence. "
    )