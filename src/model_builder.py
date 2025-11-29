"""
This module contains all functions used to create a TensorFlow model for defect detection and localization.
The model uses transfer learning with a configurable backbone and custom heads for classification and segmentation.
The model architecture supports different backbone networks and mask head types (FCN-lite and U-Net-lite).
"""

# Imports
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtTiny, EfficientNetV2B3, EfficientNetV2S, MobileNetV3Large
from typing import Tuple

# Backbone class map
backbone_class_map = {
    'ConvNeXtTiny': ConvNeXtTiny,
    'EfficientNetV2B3': EfficientNetV2B3,
    'EfficientNetV2S': EfficientNetV2S,
    'MobileNetV3Large': MobileNetV3Large
}


# Function to create the base model
def create_base_model(run_cfg, backbone_class_map) -> tf.keras.Model:
    """
    Creates the base model (backbone) for transfer learning.
    To be used in the model_creation function.
    Args:
        run_cfg (OmegaConf object): Contains the information of the global YAML config files plus those specific to the experiment begin run where the model parameters are defined.
        input_tensor (tf.Tensor): The input tensor for the model, which is created on the model_creation function.
        backbone_class_map (dict): A mapping of backbone class names to their corresponding TensorFlow Keras application classes, found on section 1.2 Global Settings.
    Returns:
        base_model: A TensorFlow Keras model instance representing the base model without the top classification layers.
    """
    # Validate backbone configuration exists
    if run_cfg.base not in run_cfg.backbone.keys():
        raise ValueError(f'Unsupported backbone {run_cfg.base}. Supported backbones are: {list(run_cfg.backbone.keys())}')
    
    # Get the backbone class from the configuration
    backbone_cls = backbone_class_map[run_cfg.backbone[run_cfg.base].class_name]
    
    # Include model's preprocessing layer
    extra_args = {}
    if run_cfg.backbone[run_cfg.base].include_preprocessing:
        extra_args['include_preprocessing'] = True

    # Create the base model
    base_model = backbone_cls(
        include_top=False,
        weights='imagenet', pooling=None, **extra_args)
    # Assign a config based name to the base model so that it can be identified later for freezing/unfreezing
    name = run_cfg.base

    # U-Net-Lite style head
    if run_cfg.mask_head.type == 'U_Net_Lite':
        # Extract skip connections based on layer names defined in config
        skip_layer_names = run_cfg.backbone[run_cfg.base]['skip_layer_names']
        outputs = [base_model.get_layer(name).output for name in skip_layer_names]
        # Append skips to the backbone output map
        outputs.append(base_model.output)
        return tf.keras.Model(inputs=base_model.input, outputs=outputs, name=name)
    
    # FCN-Lite style head
    elif run_cfg.mask_head.type == 'FCN_Lite':
        # Use directly the backbone output feature map
        return tf.keras.Model(inputs=base_model.input, outputs=base_model.output, name=name)
    
    else:
        raise ValueError(f'Unsupported mask head type {run_cfg.mask_head.type}.')
    

# Auxiliary block: SeparableConv2D + BatchNorm + ReLU (+ optional SpatialDropout2D)
def sep_conv_bn_relu(x, filters, name, kernel_size=3, dropout_rate=0.0) -> tf.Tensor:
    """
    SeparableConv2D + BatchNorm + ReLU (+ optional SpatialDropout2D) block.
    Used in both FCN-lite and U-Net-lite mask heads.
    Args:
        x: Input tensor.
        filters: Number of filters for SeparableConv2D.
        kernel_size: Kernel size for SeparableConv2D.
        name: Base name for the layers.
        dropout_rate: Dropout rate for SpatialDropout2D (0.0 = no dropout).
    Returns:
        Output tensor after applying the block.
    """
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding='same', use_bias=False, name=f'{name}_sepconv')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_bn')(x)
    x = tf.keras.layers.ReLU(name=f'{name}_relu')(x)
    if dropout_rate and dropout_rate > 0.0:
        x = tf.keras.layers.SpatialDropout2D(dropout_rate, name=f'{name}_drop')(x)
    return x


# Helper function to align tensor x to the spatial size of reference tensor ref
def align_like_ref(inputs) -> tf.Tensor:
    x, ref = inputs
    return tf.image.resize(x, tf.shape(ref)[1:3], method='bilinear')


# Output shape helper function for align_like_ref
def align_like_ref_output_shape(input_shapes) -> Tuple[int, int, int, int]:
    x_shape, ref_shape = input_shapes
    return (x_shape[0], ref_shape[1], ref_shape[2], x_shape[3])


# Lambda layer for mask-guided pooling
def mask_guided_pooling(inputs) -> tf.Tensor:
    """
    Uses the resized localization mask to pool features from the feature map.
    Enables classification head to focus on defect regions.
    """
    feat_map, mask = inputs  # feat_map: [B, H, W, C], mask: [B, H, W, 1]
    weighted_features = feat_map * mask  # [B, H, W, C]
    mask_sum = tf.reduce_sum(mask, axis=[1, 2], keepdims=True) + 1e-8  # [B,1,1,1]
    pooled = tf.reduce_sum(weighted_features, axis=[1, 2]) / tf.squeeze(mask_sum, axis=[1,2])  # [B, C]
    return pooled


# Output shape helper function for mask_guided_pooling
def mask_guided_pooling_output_shape(input_shapes) -> Tuple[int, int]:
    # feat_map: [batch, H, W, C], mask: [batch, H, W, 1]
    # Output: [batch, C]
    feat_map_shape, mask_shape = input_shapes
    return (feat_map_shape[0], feat_map_shape[3])


# Lambda layer to stop gradient flow
def stop_gradient_fn(x) -> tf.Tensor:
    return tf.stop_gradient(x)


# Function to define FCN-Lite style mask head
def build_mask_head_fcn_lite(feat, run_cfg) -> tf.Tensor:
    """
    Lightweight FCN-style decoder that upsamples to input size.
    Uses only the final backbone feature map (no top, no global pooling).
    Args:
        feat: Feature map tensor from the backbone.
        run_cfg: Configuration object with mask head parameters.
    Returns:
        Output mask tensor.
    """
    filters = int(getattr(run_cfg.mask_head, 'FCN_filters', 256))
    blocks  = int(getattr(run_cfg.mask_head, 'FCN_blocks', 2))
    dropout = float(getattr(run_cfg.mask_head, 'dropout', 0.0))
    target_size = tuple(run_cfg.input_size)  # (H, W)

    x = feat
    for i in range(blocks):
        x = sep_conv_bn_relu(x, filters, name=f'fcnlite_block{i+1}', dropout_rate=dropout)

    # Upsample to target size with bilinear resize
    x = tf.keras.layers.Resizing(target_size[0], target_size[1], interpolation='bilinear', name='fnclite_resize_to_input')(x)

    # filter = 'number of classes' = 1 channel for binary mask (defect vs no defect)
    # kernel_size = 1 to turn features per pixel into probability per pixel
    # activation = 'sigmoid' for binary mask, 'softmax' for multi-class mask
    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='mask')(x)

    return out


# Function to define U-Net-Lite style mask head
def build_mask_head_unet_lite(skips, run_cfg) -> tf.Tensor:
    """
    U-Net-Lite decoder that expects a dict of skip feature maps:
      skips['C5'] = deepest (lowest resolution)
      skips['C4'], ['C3'], ['C2'] = higher-resolution encoder outputs
    Args:
        skips: Dict of feature map tensors from the backbone.
        run_cfg: Configuration object with mask head parameters.
    Returns:
        Output mask tensor.
    """
    base_filters = int(getattr(run_cfg.mask_head, 'UNET_base_filters', 128))
    dropout      = float(getattr(run_cfg.mask_head, 'dropout', 0.0))
    target_size  = tuple(run_cfg.input_size)

    # Expected keys ordered deepest->shallowest for decoding
    order = ['C5', 'C4', 'C3', 'C2']
    feats = [skips[k] for k in order if k in skips]
    if len(feats) < 2:
        raise ValueError('U-Net-lite requires at least two skip stages (e.g., C5 and C4).')

    x = feats[0]  # start from deepest
    for i, skip in enumerate(feats[1:], start=1):
        x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear', name=f'unet_up{i}')(x)
        # If shapes mismatch due to stride differences, resize x to skip spatial size
        x = tf.keras.layers.Lambda(
            align_like_ref,
            output_shape=align_like_ref_output_shape,
            name=f'unet_align{i}'
        )([x, skip])
        x = tf.keras.layers.Concatenate(name=f'unet_concat{i}')([x, skip])
        x = sep_conv_bn_relu(x, base_filters // (2**(i-1)), name=f'unet_dec{i}_a', dropout_rate=dropout)
        x = sep_conv_bn_relu(x, base_filters // (2**(i-1)), name=f'unet_dec{i}_b', dropout_rate=dropout)

    # Final resize to exact input size (in case of odd sizes)
    x = tf.keras.layers.Resizing(target_size[0], target_size[1], interpolation='bilinear', name='unet_resize_to_input')(x)
    out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='mask')(x)
    return out


# Function to create the complete model
def create_model(run_cfg, num_classes = 6, backbone_class_map=backbone_class_map) -> tf.keras.Model:
    """
    Creates the complete model (backbone + heads) ready for training.
    Args:
        run_cfg (OmegaConf object): Contains the information of the global YAML config files plus those specific to the experiment begin run where the model parameters are defined.
        backbone_class_map (dict): A mapping of backbone class names to their corresponding TensorFlow Keras application classes, found on section 1.2 Global Settings.
        num_classes (int): The number of classes for the classification head output layer. Obtained dynamically in section 1.3 Metadata.
    Returns:
        model: A TensorFlow Keras model instance representing the complete model ready for training.
    """
    # Input layer
    target_size = tuple(run_cfg.input_size)
    input_image = tf.keras.Input(shape=(int(target_size[0]), int(target_size[1]), 3), name='image') # (height, width, channels)

    # Base model
    base_model = create_base_model(run_cfg, backbone_class_map)
    
    # Note on training=False:
    # Force the backbone to run in inference mode (training=False) so layers with
    # training/inference differences behave stably:
    #   - BatchNorm: moving stats will not be updated.
    #   - StochasticDepth/DropPath (if present): disabled.
    # This preserves pretrained normalization and disables stochastic layers, which is
    # desirable for our small-dataset.
    # Note: When fine-tuning, unfreezing the backbone (trainable=True) will still allow
    # BN affine params (gamma/beta) to train unless you also set BN layers' trainable=False.
    # Moving stats remain frozen because we call the backbone with training=False.

    # LOCALIZATION HEAD (mask)
    if run_cfg.mask_head.type == 'FCN_Lite':
        feat_map = base_model(input_image, training=False)  # shape ~ (H', W', C)
        output_mask = build_mask_head_fcn_lite(feat_map, run_cfg)
    elif run_cfg.mask_head.type == 'U_Net_Lite':
        backbone_outputs = base_model(input_image, training=False)
        feat_map = backbone_outputs[-1]  # shape ~ (H', W', C)
        skips = backbone_outputs[:-1] 
        aliases = [f'C{idx+2}' for idx in range(len(skips))]
        skips_dict = {alias: t for alias, t in zip(aliases, skips)}
        output_mask = build_mask_head_unet_lite(skips_dict, run_cfg)
    else:
        raise ValueError(f'Unknown mask head type: {run_cfg.mask_head.type}')

    # CLASSIFICATION HEAD (label)
    # Mask-guided pooling
    if run_cfg.label_head.get("independent_FCN", False):
        x = feat_map
        for i in range(run_cfg.label_head.FCN_blocks):
            x = sep_conv_bn_relu(x, run_cfg.label_head.FCN_filters, name=f'class_fcnlite_block{i+1}', dropout_rate=run_cfg.label_head.dropout)
        mask_for_cls = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='classification_mask')(x)

    else:
        mask_for_cls = tf.keras.layers.Lambda(stop_gradient_fn, name='stop_gradient')(output_mask) # Stop gradient flow from classification head to mask head
    
    mask_down = tf.keras.layers.Resizing(feat_map.shape[1], feat_map.shape[2], interpolation='bilinear')(mask_for_cls)  # Downsample mask to match feature map size
    mask_pooled = tf.keras.layers.Lambda(mask_guided_pooling,
                                         output_shape=mask_guided_pooling_output_shape,
                                         name='mask_guided_pooling'
                                         )([feat_map, mask_down]) # output_shape=(feat_map[0], feat_map[3])
    # Global average pooling
    global_features = tf.keras.layers.GlobalAveragePooling2D()(feat_map)
    # Concatenate
    final_features = tf.keras.layers.Concatenate()([global_features, mask_pooled])
    # Feed to dense layers for classification
    x_cls = tf.keras.layers.Dense(run_cfg.label_head.dense_units[0], activation='relu')(final_features)
    x_cls = tf.keras.layers.BatchNormalization()(x_cls)
    if run_cfg.label_head.dropout and run_cfg.label_head.dropout > 0.0:
        x_cls = tf.keras.layers.Dropout(run_cfg.label_head.dropout)(x_cls)
    x_cls = tf.keras.layers.Dense(run_cfg.label_head.dense_units[1], activation='relu')(x_cls)
    x_cls = tf.keras.layers.BatchNormalization()(x_cls)
    if run_cfg.label_head.dropout and run_cfg.label_head.dropout > 0.0:
        x_cls = tf.keras.layers.Dropout(run_cfg.label_head.dropout)(x_cls)
    output_label = tf.keras.layers.Dense(num_classes, activation='softmax', name='label')(x_cls)

    # Combine inputs and outputs into a model
    model = tf.keras.Model(inputs=input_image, outputs={'label': output_label, 'mask': output_mask,})
    model.backbone = base_model # Attach handle as reference for freezing/unfreezing it later on

    return model