from tensorflow import keras

IMG_SIZE = 384
IMG_SIZE_2 = 256
NUM_CLASSES = 14
BATCH_SIZE = 8
NUM_EPOCHS = 100

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

################## TRANSFORMERS ##########################
trans_learning_rate = 0.0001
trans_weight_decay = 1e-5
trans_image_size = 456  # We'll resize input images to this size
trans_patch_size = 128  # Size of the patches to be extract from the input images
trans_patch_size_2 = 48  # Size of the patches to be extract from the input images
trans_num_patches = (IMG_SIZE // trans_patch_size) ** 2
trans_num_patches_2 = 1
trans_projection_dim = 16
trans_num_heads = 32
trans_transformer_units = [
    trans_projection_dim * 2,
    trans_projection_dim,
]  # Size of the transformer layers
trans_transformer_layers = 8
trans_mlp_head_units = [256, 128]  # Size of the dense layers of the final classifier


