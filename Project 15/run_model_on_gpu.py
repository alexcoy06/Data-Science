
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def load_train(path):
    # Define the ImageDataGenerator for training
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    return train_gen_flow


def load_test(path):
    # Define the ImageDataGenerator for validation
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    test_gen_flow = test_datagen.flow_from_directory(
        path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    return test_gen_flow


def create_model(input_shape=(224, 224, 3)):
    model = Sequential([
        ResNet50(weights='imagenet', include_top=False, input_shape=input_shape),
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mae', metrics=['mae'])
    return model


def train_model(model, train_data, test_data, batch_size=32, epochs=20, steps_per_epoch=None, validation_steps=None):
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    return model


