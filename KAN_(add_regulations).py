import tensorflow as tf
from keras.layers import Layer, BatchNormalization, Input
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np

class KANSplineLayer(Layer):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 regularize_activation=1.0, regularize_entropy=1.0, **kwargs):
        super(KANSplineLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.regularize_activation = regularize_activation
        self.regularize_entropy = regularize_entropy

    def build(self, input_shape):
        self.base_weight = self.add_weight(
            shape=(self.in_features, self.out_features),
            initializer='he_normal',
            trainable=True,
            name="base_weight"
        )
        
        self.spline_weight = self.add_weight(
            shape=(self.out_features, self.in_features, self.grid_size + self.spline_order + 1),
            initializer='he_normal',
            trainable=True,
            name="spline_weight"
        )
        
        self.spline_scaler = self.add_weight(
            shape=(self.out_features, self.in_features),
            initializer='ones',
            trainable=True,
            name="spline_scaler"
        )
        
        grid_min, grid_max = -1.0, 1.0
        knots = np.linspace(grid_min, grid_max, self.grid_size + self.spline_order + 1)
        self.knots = K.constant(knots, dtype='float32')  # 形状：(num_knots,)
        
        self.bn_base = BatchNormalization()
        self.bn_spline = BatchNormalization()
        
        super(KANSplineLayer, self).build(input_shape)

    def b_splines(self, x):
        x_exp = K.expand_dims(x, axis=-1)  # (batch_size * spatial_size, in_features, 1)
        knots_exp = K.reshape(self.knots, (1, 1, -1))  # (1, 1, num_knots)
        
        abs_diff = K.abs(x_exp - knots_exp)  # (batch_size * spatial_size, in_features, num_knots)
        one_minus_abs = 1.0 - abs_diff  # (batch_size * spatial_size, in_features, num_knots)
        basis = K.maximum(0.0, one_minus_abs)  # (batch_size * spatial_size, in_features, num_knots)
        
        return basis  # (batch_size * spatial_size, in_features, num_knots)

    def call(self, x, training=False):
        input_shape = K.shape(x)
        spatial_dims = input_shape[1:-1]
        batch_size = input_shape[0]
        spatial_size = spatial_dims[0] * spatial_dims[1]

        x_reshaped = K.reshape(x, (-1, self.in_features))  # (batch_size * spatial_size, in_features)
        
        base_output = K.dot(x_reshaped, self.base_weight)  # (batch_size * spatial_size, out_features)
        base_output = self.bn_base(base_output, training=training)
        base_output = tf.keras.activations.silu(base_output)  # 非线性变换
        
        spline_weight_scaled = self.spline_weight * K.expand_dims(self.spline_scaler, axis=-1)  # (out, in, num_knots)
        spline_weight_flat = K.reshape(
            spline_weight_scaled,
            (self.out_features, self.in_features * (self.grid_size + self.spline_order + 1))
        )  # (out_features, in_features * num_knots)
        
        x_min = K.min(x_reshaped, axis=0, keepdims=True)  # (1, in_features)
        x_max = K.max(x_reshaped, axis=0, keepdims=True)  # (1, in_features)
        x_normalized = (x_reshaped - x_min) / (x_max - x_min + K.epsilon())  # (batch_size * spatial_size, in_features)
        
        spline_basis = self.b_splines(x_normalized)  # (batch_size * spatial_size, in_features, num_knots)
        spline_basis_flat = K.reshape(
            spline_basis,
            (K.shape(x_reshaped)[0], self.in_features * (self.grid_size + self.spline_order + 1))
        )  # (batch_size * spatial_size, in_features * num_knots)
        
        spline_output = K.dot(spline_basis_flat, K.transpose(spline_weight_flat))  # (batch_size * spatial_size, out_features)
        spline_output = self.bn_spline(spline_output, training=training)
        
        output = base_output + spline_output  # (batch_size * spatial_size, out_features)
        output_reshaped = K.reshape(output, (batch_size, spatial_dims[0], spatial_dims[1], self.out_features))  # (batch_size, H, W, out_features)
        
        self.add_regularization_losses(x_reshaped)
        
        return output_reshaped

    def add_regularization_losses(self, x_reshaped):
        l1_fake = K.mean(K.abs(self.spline_weight), axis=-1)  # (out_features, in_features)
        regularization_loss_activation = K.sum(l1_fake)  

        sum_l1 = K.sum(l1_fake) + K.epsilon()
        p = l1_fake / sum_l1  # (out_features, in_features)
        entropy = -K.sum(p * K.log(p + K.epsilon()))  

        regularization_loss = (self.regularize_activation * regularization_loss_activation +
                               self.regularize_entropy * entropy)
        
        self.add_loss(regularization_loss)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_features,)

    def get_config(self):
        config = super(KANSplineLayer, self).get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'regularize_activation': self.regularize_activation,
            'regularize_entropy': self.regularize_entropy
        })
        return config

