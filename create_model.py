"""
Creates keras model to control a car.
"""
import cv2
import random
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from sklearn.model_selection import train_test_split

DATA_DIRECTORY = './data/data_downloaded/'
DATA_FILE = './data/data_downloaded/driving_log.csv'


def preprocess(image):
    """
    Does an image preprocessing.

    :param np.ndarray image: Image to be preprocessed.

    :rtype: np.ndarray
    """
    cropped = image[50:130, :, :]
    resized = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA) / 255.
    return resized


def show_gallery(images, n_rows, n_cols):
    """
    Shows a gallery of images.
    """
    def iter_axes(ax):
        for row in ax:
            for col in row:
                yield col

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8, 9))
    fig.tight_layout()

    axes_iterator = iter_axes(axes)
    for image, subplot in zip(images, axes_iterator):
        subplot.axis('off')
        subplot.set_title(image.shape)
        subplot.imshow(image)
    for remaining_subplot in axes_iterator:
        remaining_subplot.axis('off')
    plt.show()


def summarize(data):
    """
    Plots data
    """
    sample_data_size = 10
    random_images = random.sample(data, sample_data_size)
    random_preprocessed_images = []
    for image_path in random_images:
        image = cv2.imread(DATA_DIRECTORY + image_path).astype(np.float32)
        random_preprocessed_images.append(preprocess(image))

    show_gallery(random_preprocessed_images, 5, 2)


def NvidiaModel():
    """
    Creates nvidia model for steering a car.

    :return:
    """
    nvidia_model = models.Sequential()

    nvidia_model.add(layers.Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', batch_input_shape=(None, 66, 200, 3)))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Flatten())

    nvidia_model.add(layers.Dense(1164))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Dense(100))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Dense(50))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Dense(10))
    nvidia_model.add(layers.Activation('relu'))

    nvidia_model.add(layers.Dense(1))

    return nvidia_model


def augment_brightness(image, number_of_new_images):
    """
    Generates additional data by changing image brightness.
    """
    additional_images = []
    for i in range(number_of_new_images):
        generated_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        generated_image[:, :, 2] = generated_image[:, :, 2] * (0.25 + np.random.uniform())
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_HSV2BGR)
        additional_images.append(preprocess(generated_image))

    return additional_images


def augment_by_random_image_shift(image, steer, number_of_additional_images):
    additional_images = []
    additional_steering = []

    for _ in range(number_of_additional_images):
        horizontal_shift = np.random.randint(-30, 30)
        shifted_steering_angle = steer + horizontal_shift * 0.008
        translation_matrix = np.float32([[1, 0, horizontal_shift], [0, 1, 0]])
        shifted_image = cv2.warpAffine(image, translation_matrix, (320, 160))

        additional_images.append(preprocess(shifted_image))
        additional_steering.append(shifted_steering_angle)

    return additional_images, additional_steering


def generate_additional_data(image, steering):
    """
    Generates additional image data.
    """
    additional_images = []
    additional_steering_angles = []

    brightness_changes = 6
    image_shift_changes = 10

    brightness_changed_images = augment_brightness(image, brightness_changes)
    additional_images.extend(brightness_changed_images)
    additional_steering_angles.extend([steering] * brightness_changes)

    shifted_images, shifted_angles = augment_by_random_image_shift(image, steering, image_shift_changes)
    additional_images.extend(shifted_images)
    additional_steering_angles.extend(shifted_angles)

    return additional_images, additional_steering_angles


def generate_train_data(train_data, batch_size):
    """
    Train data generator.
    """
    while True:

        batch_images = []
        batch_steering = []
        shuffled = train_data.reindex(np.random.permutation(train_data.index))
        for row in shuffled.iterrows():
            center_image = cv2.imread(DATA_DIRECTORY + row[1]['center'])
            left_image = cv2.imread(DATA_DIRECTORY + row[1]['left'].strip())
            right_image = cv2.imread(DATA_DIRECTORY + row[1]['right'].strip())

            batch_images.append(preprocess(center_image))
            batch_steering.append(row[1]['steering'])

            batch_images.append(preprocess(left_image))
            left_camera_steer = row[1]['steering'] + 0.25
            batch_steering.append(left_camera_steer)

            batch_images.append(preprocess(right_image))
            right_camera_steer = row[1]['steering'] - 0.25
            batch_steering.append(right_camera_steer)

            flipped_image = cv2.flip(center_image, 1)
            flipped_steering = -row[1]['steering']
            batch_images.append(preprocess(flipped_image))
            batch_steering.append(flipped_steering)

            images = [center_image, left_image, right_image, flipped_image]
            steering_angels = [row[1]['steering'], left_camera_steer, right_camera_steer, flipped_steering]
            for an_image, steering in zip(images, steering_angels):
                augmented_images, augmented_steering = generate_additional_data(an_image, steering)

                batch_images.extend(augmented_images)
                batch_steering.extend(augmented_steering)

            if len(batch_steering) >= batch_size:
                yield np.asarray(batch_images, dtype=np.float32), np.asarray(batch_steering, dtype=np.float32)
                batch_images = []
                batch_steering = []


if __name__ == '__main__':

    data = pd.read_csv(DATA_FILE)
    train_data, val_data = train_test_split(data, test_size=0.0)

    # summarize(list(train_data['center']))

    model = NvidiaModel()
    model.compile(optimizers.Adam(lr=0.0001), loss='mse')

    train_samples_per_epoch = len(train_data) * 14
    batch_size = 512
    number_of_epochs = 5
    model.fit_generator(
        generate_train_data(train_data, batch_size),
        train_samples_per_epoch,
        number_of_epochs
    )
    model_architecture = model.save('behavioral_cloning_model.h5')
