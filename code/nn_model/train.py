from pickle import dump

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from code import params
from data_processing import prepare_dataset


def train(model):
    train_df, test_df, all_labels = prepare_dataset()

    core_idg = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.1,
        width_shift_range=0.1,
        rotation_range=10,
        shear_range=0.1,
        fill_mode='reflect',
        zoom_range=0.2,
        validation_split=0.2)

    train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='newLabel',
                                             class_mode='categorical',
                                             classes=all_labels,
                                             target_size=(params.IMG_SIZE, params.IMG_SIZE),
                                             color_mode='rgb',
                                             batch_size=8,
                                             subset='training', )

    valid_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='newLabel',
                                             class_mode='categorical',
                                             classes=all_labels,
                                             target_size=(params.IMG_SIZE, params.IMG_SIZE),
                                             color_mode='rgb',
                                             batch_size=2,
                                             subset='validation')

    weight_path = "outputs/{}_weights.best.hdf5".format('xray_class')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_auc', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=0)

    callbacks_list = [checkpoint, reduce_lr]
    # model.load_weights(weight_path)
    history = model.fit(train_gen,
                        epochs=100,
                        steps_per_epoch=1000,
                        validation_data=valid_gen,
                        validation_steps=valid_gen.samples // 2,
                        callbacks=callbacks_list)
    with open(f'outputs/history.txt',
              'wb') as handle:  # saving the history of the model trained for another 50 Epochs
        dump(history.history, handle)

    model.save_weights("outputs/{}_weights.last.hdf5".format('xray_class'))
