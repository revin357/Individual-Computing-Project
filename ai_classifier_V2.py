from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
from PIL import Image as Image
import split_folders

image_shape = (224,224)

epochs = 25


def split_dataset():
    split_folders.ratio(input='./Dataset/flowers', output='./Dataset/Split_Dataset', seed=1337, ratio=(.65, .35))


def prepare_datasets():
    global image_batch, training_image_dataset, val_image_dataset
    training_data_root = './Dataset/Split_dataset/train'
    val_data_root = './Dataset/Split_Dataset/val'

    training_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    training_image_dataset = training_image_generator.flow_from_directory(str(training_data_root),
                                                                          target_size=image_shape,
                                                                          )
    val_image_dataset = val_image_generator.flow_from_directory(str(val_data_root), target_size=image_shape,
                                                                )

    for image_batch, label_batch in training_image_dataset:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    for image_batch, label_batch in val_image_dataset:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break


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


def create_resnet_model(epoch_range):
    resnet_base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    feature_batch = resnet_base_model(image_batch)
    print(feature_batch.shape)

    resnet_base_model.summary()

    fine_tune_at = 88

    for layer in resnet_base_model.layers[:fine_tune_at]:
        layer.trainable = False

    #resnet_base_model.trainable = False

    resnet_model = tf.keras.Sequential([
        resnet_base_model,
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='elu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes, activation='softmax')
    ])

    resnet_model.summary()

    resnet_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc']
    )

    create_checkpoint_callback('Res', epoch_range)

    steps_per_epoch = np.ceil(training_image_dataset.samples / training_image_dataset.batch_size)

    resnet_history = resnet_model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        callbacks=[batch_stats, cp_callback]
    )

    resnet_model.save('./ResNet/Models/Model_{}/ResNet_Model.h5'.format(epoch_range))

    res_acc = resnet_history.history['acc']
    res_val_acc = resnet_history.history['val_acc']

    res_loss = resnet_history.history['loss']
    res_val_loss = resnet_history.history['val_loss']

    epochs_range = range(epochs)

    if os.path.exists('./ResNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range)):
        pass
    else:
        os.mkdir('./ResNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, res_acc, label='Training Accuracy')
    plt.plot(epochs_range, res_val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig('./ResNet/Plots/Epochs_{}-{}/ResNet_Accuracy_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, res_loss, label='Training Loss')
    plt.plot(epochs_range, res_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./ResNet/Plots/Epochs_{}-{}/ResNet_Loss_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig('./ResNet/Plots/Epochs_{}-{}/ResNet_Loss_batchStats.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig('./ResNet/Plots/Epochs_{}-{}/ResNet_Accuracy_batchStats.png'.format(epoch_range-epochs, epoch_range))


def create_mobilenet_model(epoch_range):
    mobilenet_base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    feature_batch = mobilenet_base_model(image_batch)
    print(feature_batch.shape)

    mobilenet_base_model.summary()

    fine_tune_at = 78

    for layer in mobilenet_base_model.layers[:fine_tune_at]:
        layer.trainable = False

    #mobilenet_base_model.trainable = False

    mobilenet_model = tf.keras.Sequential([
        mobilenet_base_model,
        tf.keras.layers.Conv2D(16,3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes)
    ])

    mobilenet_model.summary()

    mobilenet_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc']
    )

    create_checkpoint_callback('Mobile', epoch_range)

    steps_per_epoch = np.ceil(training_image_dataset.samples / training_image_dataset.batch_size)

    mobilenet_history = mobilenet_model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        callbacks=[batch_stats, cp_callback]
    )

    mobilenet_model.save('./MobileNet/Models/Model_{}/MobileNet_Model.h5'.format(epoch_range))

    if os.path.exists('./MobileNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range)):
        pass
    else:
        os.mkdir('./MobileNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range))

    mobile_acc = mobilenet_history.history['acc']
    mobile_val_acc = mobilenet_history.history['val_acc']

    mobile_loss = mobilenet_history.history['loss']
    mobile_val_loss = mobilenet_history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, mobile_acc, label='Training Accuracy')
    plt.plot(epochs_range, mobile_val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig('./MobileNet/Plots/Epochs_{}-{}/MobileNet_Accuracy_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, mobile_loss, label='Training Loss')
    plt.plot(epochs_range, mobile_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./MobileNet/Plots/Epochs_{}-{}/MobileNet_Loss_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig('./MobileNet/Plots/Epochs_{}-{}/Mobile_Loss_batchStats.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig('./MobileNet/Plots/Epochs_{}-{}/Mobile_Accuracy_batchStats.png'.format(epoch_range-epochs, epoch_range))




def create_densenet_model(epoch_range):
    densenet_base_model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    feature_batch = densenet_base_model(image_batch)
    print(feature_batch.shape)

    densenet_base_model.summary()

    fine_tune_at = 214

    for layer in densenet_base_model.layers[:fine_tune_at]:
        layer.trainable = False

    #densenet_base_model.trainable = False

    densenet_model = tf.keras.Sequential([
        densenet_base_model,
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(training_image_dataset.num_classes)
    ])

    densenet_model.summary()

    densenet_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['acc']
    )

    create_checkpoint_callback('Dense', epoch_range)

    steps_per_epoch = np.ceil(training_image_dataset.samples / training_image_dataset.batch_size)

    densenet_history = densenet_model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        callbacks=[batch_stats, cp_callback]
    )

    densenet_model.save('./DenseNet/Models/Model_{}/DenseNet_Model.h5'.format(epoch_range))

    if os.path.exists('./DenseNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range)):
        pass
    else:
        os.mkdir('./DenseNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range))

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
    plt.savefig('./DenseNet/Plots/Epochs_{}-{}/DenseNet_Accuracy_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dense_loss, label='Training Loss')
    plt.plot(epochs_range, dense_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./DenseNet/Plots/Epochs_{}-{}/DenseNet_Loss_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig('./DenseNet/Plots/Epochs_{}-{}/Dense_Loss_batchStats.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig('./DenseNet/Plots/Epochs_{}-{}/Dense_Accuracy_batchStats.png'.format(epoch_range-epochs, epoch_range))


def train_resnet_model(epoch_range):
    resnet_model = tf.keras.models.load_model('./ResNet/Models/Model_{}/ResNet_Model.h5'.format(epoch_range-epochs))

    resnet_model.summary()

    create_checkpoint_callback('Res', epoch_range)

    steps_per_epoch = np.ceil(training_image_dataset.samples / training_image_dataset.batch_size)

    resnet_history = resnet_model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        callbacks=[batch_stats, cp_callback]
    )

    resnet_model.save('./ResNet/Models/Model_{}/ResNet_Model.h5'.format(epoch_range))

    if os.path.exists('./ResNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range)):
        pass
    else:
        os.mkdir('./ResNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range))

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
    plt.savefig('./ResNet/Plots/Epochs_{}-{}/ResNet_Accuracy_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, res_loss, label='Training Loss')
    plt.plot(epochs_range, res_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./ResNet/Plots/Epochs_{}-{}/ResNet_Loss_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig('./ResNet/Plots/Epochs_{}-{}/ResNet_Loss_batchStats.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig('./ResNet/Plots/Epochs_{}-{}/ResNet_Accuracy_batchStats.png'.format(epoch_range-epochs, epoch_range))


def train_mobilenet_model(epoch_range):
    mobilenet_model = tf.keras.models.load_model('./MobileNet/Models/Model_{}/MobileNet_Model.h5'.format(epoch_range-epochs))

    mobilenet_model.summary()

    create_checkpoint_callback('Mobile', epoch_range)

    steps_per_epoch = np.ceil(training_image_dataset.samples / training_image_dataset.batch_size)
    val_steps = np.ceil(val_image_dataset.samples / val_image_dataset.batch_size)

    mobilenet_history = mobilenet_model.fit(
        training_image_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_image_dataset,
        validation_steps=val_steps,
        callbacks=[batch_stats, cp_callback]
    )

    mobilenet_model.save('./MobileNet/Models/Model_{}/MobileNet_Model.h5'.format(epoch_range))

    if os.path.exists('./MobileNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range)):
        pass
    else:
        os.mkdir('./MobileNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range))

    mobile_acc = mobilenet_history.history['acc']
    mobile_val_acc = mobilenet_history.history['val_acc']

    mobile_loss = mobilenet_history.history['loss']
    mobile_val_loss = mobilenet_history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, mobile_acc, label='Training Accuracy')
    plt.plot(epochs_range, mobile_val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig('./MobileNet/Plots/Epochs_{}-{}/MobileNet_Accuracy_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, mobile_loss, label='Training Loss')
    plt.plot(epochs_range, mobile_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./MobileNet/Plots/Epochs_{}-{}/MobileNet_Loss_Epochs.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig('./MobileNet/Plots/Epochs_{}-{}/Mobile_Loss_batchStats.png'.format(epoch_range-epochs, epoch_range))

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig('./MobileNet/Plots/Epochs_{}-{}/Mobile_Accuracy_batchStats.png'.format(epoch_range-epochs, epoch_range))


def train_densenet_model(epoch_range):
    densenet_model = tf.keras.models.load_model('./DenseNet/Models/Model_{}/DenseNet_Model.h5'.format(epoch_range-epochs))

    densenet_model.summary()

    create_checkpoint_callback('Dense', epoch_range)

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

    densenet_model.save('./DenseNet/Models/Model_{}/DenseNet_Model.h5'.format(epoch_range))

    if os.path.exists('./DenseNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range)):
        pass
    else:
        os.mkdir('./DenseNet/Plots/Epochs_{}-{}'.format(epoch_range-epochs, epoch_range))

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
    plt.savefig('./DenseNet/Plots/Epochs_{}-{}/DenseNet_Accuracy_Epochs.png'.format(epoch_range-epochs,
                                                                                    epoch_range))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dense_loss, label='Training Loss')
    plt.plot(epochs_range, dense_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./DenseNet/Plots/Epochs_{}-{}/DenseNet_Loss_Epochs.png'.format(epoch_range-epochs,
                                                                                epoch_range))

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.ylim([0, 2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig('./DenseNet/Plots/Epochs_{}-{}/DenseNet_Loss_batchStats.png'.format(epoch_range-epochs,
                                                                                    epoch_range))

    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Training Steps")
    plt.ylim([0, 1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig('./DenseNet/Plots/Epochs_{}-{}/DenseNet_Accuracy_batchStats.png'.format(epoch_range-epochs,
                                                                                        epoch_range))


res_epoch_range = dense_epoch_range = mobile_epoch_range = epochs

if os.path.exists('./Dataset/Split_dataset'):
    pass
else:
    split_dataset()

prepare_datasets()

create_batch_stats_callback()

if os.path.exists('./ResNet/Models/Model_{}/ResNet_Model.h5'.format(res_epoch_range)):
    res_epoch_range = res_epoch_range + epochs
else:
    create_resnet_model(res_epoch_range)
    res_epoch_range = res_epoch_range + epochs

if os.path.exists('./MobileNet/Models/Model_{}/MobileNet_Model.h5'.format(mobile_epoch_range)):
    mobile_epoch_range = mobile_epoch_range + epochs
else:
    create_mobilenet_model(mobile_epoch_range)
    mobile_epoch_range = mobile_epoch_range + epochs

if os.path.exists('./DenseNet/Models/Model_{}/DenseNet_Model.h5'.format(dense_epoch_range)):
    dense_epoch_range = dense_epoch_range + epochs
else:
    create_densenet_model(dense_epoch_range)
    dense_epoch_range = dense_epoch_range + epochs

if os.path.exists('./ResNet/Models/Model_{}/ResNet_Model.h5'.format(res_epoch_range)):
    res_epoch_range = res_epoch_range + epochs
else:
    train_resnet_model(res_epoch_range)
    res_epoch_range = res_epoch_range + epochs

if os.path.exists('./MobileNet/Models/Model_{}/MobileNet_Model.h5'.format(mobile_epoch_range)):
    mobile_epoch_range = mobile_epoch_range + epochs
else:
    train_mobilenet_model(mobile_epoch_range)
    mobile_epoch_range = mobile_epoch_range + epochs

if os.path.exists('./DenseNet/Models/Model_{}/DenseNet_Model.h5'.format(dense_epoch_range)):
    dense_epoch_range = dense_epoch_range + epochs
else:
    train_densenet_model(dense_epoch_range)
    dense_epoch_range = dense_epoch_range + epochs

if os.path.exists('./ResNet/Models/Model_{}/ResNet_Model.h5'.format(res_epoch_range)):
    res_epoch_range = res_epoch_range + epochs
else:
    train_resnet_model(res_epoch_range)
    res_epoch_range = res_epoch_range + epochs

if os.path.exists('./MobileNet/Models/Model_{}/MobileNet_Model.h5'.format(mobile_epoch_range)):
    mobile_epoch_range = mobile_epoch_range + epochs
else:
    train_mobilenet_model(mobile_epoch_range)
    mobile_epoch_range = mobile_epoch_range + epochs

if os.path.exists('./DenseNet/Models/Model_{}/DenseNet_Model.h5'.format(dense_epoch_range)):
    dense_epoch_range = dense_epoch_range + epochs
else:
    train_densenet_model(dense_epoch_range)
    dense_epoch_range = dense_epoch_range + epochs

if os.path.exists('./ResNet/Models/Model_{}/ResNet_Model.h5'.format(res_epoch_range)):
    pass
else:
    train_resnet_model(res_epoch_range)

if os.path.exists('./MobileNet/Models/Model_{}/MobileNet_Model.h5'.format(mobile_epoch_range)):
    pass
else:
    train_mobilenet_model(mobile_epoch_range)

if os.path.exists('./DenseNet/Models/Model_{}/DenseNet_Model.h5'.format(dense_epoch_range)):
    pass
else:
    train_densenet_model(dense_epoch_range)
