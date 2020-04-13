from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras

import tensorflow_hub as hub
import numpy as np
from PIL import Image as Image

image_shape = (224,224)

epochs = 25


def prepare_datasets():
    global image_batch, training_image_dataset, val_image_dataset
    training_data_root = './Dataset/Split_dataset/train'
    val_data_root = './Dataset/Split_Dataset/val'

    training_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    training_image_dataset = training_image_generator.flow_from_directory(str(training_data_root),
                                                                          target_size=image_shape)
    val_image_dataset = val_image_generator.flow_from_directory(str(val_data_root), target_size=image_shape)

    for image_batch, label_batch in training_image_dataset:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    for image_batch, label_batch in val_image_dataset:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break


def create_resnet_model():
    resnet_classifier = hub.KerasLayer(
        tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=1000),
        input_shape=(224, 224, 3))

    feature_batch = resnet_classifier(image_batch)
    print(feature_batch.shape)

    resnet_classifier.trainable = False

    resnet_model = tf.keras.Sequential([
        resnet_classifier,
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='softmax')
    ])

    resnet_model.summary()

    resnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    class CollectBatchStats(tf.keras.callbacks.Callback):
        def __init__(self):
            self.batch_losses = []
            self.batch_acc = []

        def on_train_batch_end(self, batch, logs=None):
            self.batch_losses.append(logs['loss'])
            self.batch_acc.append(logs['acc'])
            self.model.reset_metrics()

    batch_stats = CollectBatchStats()

    checkpoint_path = './ResNet/Model/Checkpoints/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    steps_per_epoch = np.ceil(training_image_dataset.samples / training_image_dataset.batch_size)
    val_steps = np.ceil(val_image_dataset.samples / val_image_dataset.batch_size)

    resnet_history = resnet_model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        validation_steps=val_steps,
        callbacks=[batch_stats, cp_callback]
    )

    resnet_model.save('./ResNet/Model/', save_format='tf')

    res_acc = resnet_history.history['acc']
    res_val_acc = resnet_history.history['val_acc']

    res_loss = resnet_history.history['loss']
    res_val_loss = resnet_history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, res_acc, label='Training Accuracy')
    plt.plot(epochs_range, res_val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig('./ResNet/Plots/ResNet_Accuracy_Epochs_25.png')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, res_loss, label='Training Loss')
    plt.plot(epochs_range, res_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./ResNet/Plots/ResNet_Loss_Epochs_25.png')

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig('./ResNet/Plots/ResNet_Loss_batchStats_25.png')

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig('./ResNet/Plots/ResNet_Accuracy_batchStats_25.png')




def create_densenet_model():
    densenet_classifier = hub.KerasLayer(
        tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=1000),
        input_shape=(224, 224, 3))

    feature_batch = densenet_classifier(image_batch)
    print(feature_batch.shape)

    densenet_classifier.trainable = False

    densenet_model = tf.keras.Sequential([
        densenet_classifier,
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='softmax')
    ])

    densenet_model.summary()

    densenet_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    class CollectBatchStats(tf.keras.callbacks.Callback):
        def __init__(self):
            self.batch_losses = []
            self.batch_acc = []

        def on_train_batch_end(self, batch, logs=None):
            self.batch_losses.append(logs['loss'])
            self.batch_acc.append(logs['acc'])
            self.model.reset_metrics()

    batch_stats = CollectBatchStats()

    checkpoint_path = './DenseNet/Model/Checkpoints/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    steps_per_epoch = np.ceil(training_image_dataset.samples / training_image_dataset.batch_size)
    val_steps = np.ceil(val_image_dataset.samples / val_image_dataset.batch_size)

    densenet_history = densenet_model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        validation_steps=val_steps,
        callbacks=[batch_stats, cp_callback]
    )

    densenet_model.save('./DenseNet/Model/', save_format='tf')

    dense_acc = densenet_history.history['acc']
    dense_val_acc = densenet_history.history['val_acc']

    dense_loss = densenet_history.history['loss']
    dense_val_loss = densenet_history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dense_acc, label='Training Accuracy')
    plt.plot(epochs_range, dense_val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig('./DenseNet/Plots/DenseNet_Accuracy_Epochs_25.png')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dense_loss, label='Training Loss')
    plt.plot(epochs_range, dense_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./DenseNet/Plots/DenseNet_Loss_Epochs_25.png')

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig('./DenseNet/Plots/Dense_Loss_batchStats_25.png')

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig('./DenseNet/Plots/Dense_Accuracy_batchStats_25.png')


prepare_datasets()
create_resnet_model()
create_densenet_model()
