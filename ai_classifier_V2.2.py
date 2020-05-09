from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
from PIL import Image as Image
import split_folders
import IPython.display as display
import pathlib
import itertools
import sklearn.metrics

import datetime

import sys

# Define the image shape for image preprocessing.
image_shape = (224,224)

# Define the number of epochs per fit function.
epochs = 25

# Define batch size for images to be split into.
batch_size = 32

learning_rate = 1e-3


# Function to split raw dataset into train and validation sets.
def split_dataset():
    # Splits raw dataset of Caltech256 into train and test sets at a ratio of 70% train and 30% test.
    split_folders.ratio(input='./Dataset/Caltech256', output='./Dataset/Caltech_Split_Dataset', seed=1337, ratio=(.70, .30))


# Function to preprocess the train and validation datasets.
def prepare_datasets():
    # Define global variables to be used outside of function.
    global image_batch, training_image_dataset, val_image_dataset, steps_per_epoch, val_steps, class_names
    # Assign training dataset root directory to variable
    training_data_root = './Dataset/Caltech_Split_Dataset/train'
    # Assign validation dataset root directory to variable
    val_data_root = './Dataset/Caltech_Split_Dataset/val'


    training_data_root = pathlib.Path(training_data_root)
    val_data_root = pathlib.Path(val_data_root)

    # Creates an array of the class names for the dataset using directory names and excluding the LICENSE.txt file
    class_names = np.array([item.name for item in training_data_root.glob('*') if item.name != "LICENSE.txt"])

    training_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    training_image_dataset = training_image_generator.flow_from_directory(str(training_data_root),
                                                                          target_size=image_shape,
                                                                          shuffle=True,
                                                                          batch_size=batch_size,
                                                                          class_mode='categorical',
                                                                          classes=list(class_names)
                                                                          )
    val_image_dataset = val_image_generator.flow_from_directory(str(val_data_root),
                                                                target_size=image_shape,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                class_mode='categorical'
                                                                )

    for image_batch, label_batch in training_image_dataset:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    for image_batch, label_batch in val_image_dataset:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    steps_per_epoch = np.ceil(training_image_dataset.samples / training_image_dataset.batch_size)
    val_steps = np.ceil(val_image_dataset.samples / val_image_dataset.batch_size)


def create_batch_stats_callback():
    global batch_stats

    class CollectBatchStats(tf.keras.callbacks.Callback):
        def __init__(self):
            self.batch_losses = []
            self.batch_acc = []

        def on_train_batch_end(self, batch, logs=None):
            self.batch_losses.append(logs['loss'])
            self.batch_acc.append(logs['acc'])
            self.model.reset_metrics()

    batch_stats = CollectBatchStats()


def create_checkpoint_callback(model_type, epoch_range):
    global checkpoint_dir, cp_callback
    checkpoint_path = './{}Net/Models/Model_{}/Checkpoints/cp.ckpt'.format(model_type, epoch_range)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


# Creates a callback for use in training that stops the training if a defined metric does not improve
def create_early_stopping_callback():
    # Define global variables to be used outside of function.
    global es_callback
    # Create early stopping callback which examines the val_loss metric of the model and
    # stops the training if the metric does not decrease by 0.01 for 3 epochs of training then restores the best weights
    # recorded during those epochs.
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=5,
        min_delta=1e-3
    )


# Function to create graphs from the training data using a models history the epoch range that has been trained and
# the model type.
def create_graphs(model_type, history, epoch_range):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    if os.path.exists('./{}Net/Plots/Epochs_{}-{}'.format(model_type, epoch_range - epochs, epoch_range)):
        pass
    else:
        os.mkdir('./{}Net/Plots/Epochs_{}-{}'.format(model_type, epoch_range - epochs, epoch_range))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.savefig('./{}Net/Plots/Epochs_{}-{}/{}Net_Accuracy_Epochs.png'.format(model_type, epoch_range - epochs,
                                                                                  epoch_range, model_type))

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(
            './{}Net/Plots/Epochs_{}-{}/{}Net_Loss_Epochs.png'.format(model_type, epoch_range - epochs, epoch_range,
                                                                      model_type))

        plt.figure()
        plt.ylabel("Loss")
        plt.xlabel("Training Steps")
        plt.ylim([0, 2])
        plt.plot(batch_stats.batch_losses)
        plt.savefig('./{}Net/Plots/Epochs_{}-{}/{}Net_Loss_batchStats.png'.format(model_type, epoch_range - epochs,
                                                                                  epoch_range, model_type))

        plt.figure()
        plt.ylabel("Accuracy")
        plt.xlabel("Training Steps")
        plt.ylim([0, 1])
        plt.plot(batch_stats.batch_acc)
        plt.savefig(
            './{}Net/Plots/Epochs_{}-{}/{}Net_Accuracy_batchStats.png'.format(model_type, epoch_range - epochs,
                                                                              epoch_range, model_type))


# Function that creates the models and runs the first round of training
def create_model(epoch_range, model_type):

    # Uses the model type parameter to define the type of feature vector model to be used at the base model.
    if model_type == 'Res':
        base_model = hub.KerasLayer(
            'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1',
            input_shape=(224,224,3),
            trainable=False
        )
    elif model_type == 'Mobile':
        base_model = hub.KerasLayer(
            'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4',
            input_shape=(224,224,3),
            trainable=False
        )
    elif model_type == "Inception":
        base_model = hub.KerasLayer(
            'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4',
            input_shape=(224,224,3),
            trainable=False
        )
    else:
        sys.exit(0)


    # Creates a Sequential model using tensorflow hub to import a pretrained model then uses transfer learning to
    # train the model on the dataset
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='softmax')
    ])

    # Displays a summary of the created model
    model.summary()

    # Compiles the model using the Adam optimizer with a learning rate of 0.0001 and uses Categorical Crossentropy to
    # calculate the loss of the model and accuracy as a metric.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['acc']
    )

    log_dir = "logs\\fit\\" + model_type + str(learning_rate) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    create_checkpoint_callback(model_type, epoch_range)

    history = model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        callbacks=[batch_stats, cp_callback, tensorboard_callback, es_callback],
        verbose=1
    )

    model.save('./{}Net/Models/Model_{}/{}Net_Model.h5'.format(model_type, epoch_range, model_type))

    #create_graphs(model_type, history, epoch_range)


def train_model(epoch_range, model_type):
    model = tf.keras.models.load_model('./{}Net/Models/Model_{}/{}Net_Model.h5'.format(model_type, epoch_range-epochs, model_type), custom_objects={'KerasLayer':hub.KerasLayer})

    model.summary()

    create_checkpoint_callback(model_type, epoch_range)

    history = model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        validation_steps=val_steps,
        callbacks=[batch_stats, cp_callback, es_callback]
    )

    model.save('./{}Net/Models/Model_{}/{}Net_Model.h5'.format(model_type, epoch_range, model_type))

    create_graphs(model_type, history, epoch_range)


res_epoch_range = dense_epoch_range = mobile_epoch_range = epochs

if os.path.exists('./Dataset/Caltech_Split_dataset'):
    pass
else:
    split_dataset()

prepare_datasets()

create_batch_stats_callback()

create_early_stopping_callback()

if os.path.exists('./ResNet/Models/Model_{}/ResNet_Model.h5'.format(res_epoch_range)):
    res_epoch_range = res_epoch_range + epochs
else:
    create_model(res_epoch_range, 'Res')
    res_epoch_range = res_epoch_range + epochs

if os.path.exists('./MobileNet/Models/Model_{}/MobileNet_Model.h5'.format(mobile_epoch_range)):
    mobile_epoch_range = mobile_epoch_range + epochs
else:
    create_model(mobile_epoch_range, 'Mobile')
    mobile_epoch_range = mobile_epoch_range + epochs

if os.path.exists('./InceptionNet/Models/Model_{}/InceptionNet_Model.h5'.format(dense_epoch_range)):
    dense_epoch_range = dense_epoch_range + epochs
else:
    create_model(dense_epoch_range, 'Inception')
    dense_epoch_range = dense_epoch_range + epochs
