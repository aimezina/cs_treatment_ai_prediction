import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.applications.inception_v3 import InceptionV3

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Dense

from help_functions import mlp, Patches, PatchEncoder, inception_module
from losses import AsymmetricLossOptimized
from code.params import IMG_SIZE, NUM_CLASSES, METRICS, trans_learning_rate, trans_weight_decay, \
    trans_projection_dim, trans_patch_size_2, trans_transformer_layers, \
    trans_num_heads, trans_transformer_units, trans_mlp_head_units, trans_num_patches_2


def transformers():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    model2 = InceptionV3(include_top=False)(inputs)
    model2 = GlobalAveragePooling2D()(model2)
    model2 = mlp(model2, hidden_units=[128], dropout_rate=0.5)

    ################   Branch 1     ###########################################
    layer = inception_module(inputs, 32, 48, 64, 8, 16, 16)
    layer = inception_module(layer, 16, 24, 32, 4, 8, 8)
    layer = inception_module(layer, 8, 12, 16, 2, 4, 4)
    patches_1 = Patches(trans_patch_size_2)(layer)

    # Encode patches.
    encoded_patches_1 = PatchEncoder(trans_num_patches_2, trans_projection_dim)(patches_1)

    # Create multiple layers of the Transformer block.
    for _ in range(trans_transformer_layers):
        # Layer normalization 1.
        x1_1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_1)
        # Create a multi-head attention layer.
        attention_output_1 = layers.MultiHeadAttention(
            num_heads=trans_num_heads, key_dim=trans_projection_dim, dropout=0.1
        )(x1_1, x1_1)
        # Skip connection 1.
        x2_1 = layers.Add()([attention_output_1, encoded_patches_1])
        # Layer normalization 2.
        x3_1 = layers.LayerNormalization(epsilon=1e-6)(x2_1)
        # MLP.
        x3_1 = mlp(x3_1, hidden_units=trans_transformer_units, dropout_rate=0.2)
        # Skip connection 2.
        encoded_patches_1 = layers.Add()([x3_1, x2_1])

    # Create a [batch_size, projection_dim] tensor.
    representation_1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches_1)
    representation_1 = layers.GlobalAveragePooling1D()(representation_1)
    representation_1 = layers.Dropout(0.5)(representation_1)
    # Add MLP.
    features_1 = mlp(representation_1, hidden_units=trans_mlp_head_units, dropout_rate=0.5)
    # Classify outputs.

    common = concatenate([features_1, model2])

    common = Dense(64)(common)
    common = Dense(32)(common)

    add_final = Dense(NUM_CLASSES, activation='sigmoid')(common)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=add_final)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=trans_learning_rate, weight_decay=trans_weight_decay
    )


    model.compile(
        optimizer=optimizer,
        loss=AsymmetricLossOptimized,
        metrics=METRICS,
    )
    return model
