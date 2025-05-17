# -*- coding: utf-8 -*-
import tensorflow
import scipy.io as sio
import numpy as np
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from keras.layers import Activation
from keras.callbacks import LearningRateScheduler

from keras import backend as K, initializers
from keras.engine import Layer, InputSpec

K.set_image_dim_ordering('tf')

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Dropout
from keras.optimizers import RMSprop
from keras.layers import  Conv1D, GlobalAveragePooling2D, Multiply, BatchNormalization, Permute, Conv2D, AveragePooling2D, concatenate, activations, Bidirectional, GRU, Flatten, Concatenate
from keras.initializers import Initializer
from secondpooling import SecondOrderPooling
from tensorflow.keras import layers
from data import image_size_dict
# Import GraphConv and generate_Q
from graph_convolution import GraphConv
from GSC_utils import generate_Q

from keras.utils import np_utils
import keras

from keras.layers import MaxPooling1D
from keras.optimizers import Adam
import pandas as pd
import seaborn as sns
import time
from functions import *
from KAN import KANSplineLayer
from Rivy import RivyOptimizer
from lhadamw import LookaheadAdamW
from lhrmsprop import LookaheadRMSprop

# 1 CDC引入
class CentralDifferenceConv2D(Layer):
    def __init__(self, num_filters, kernel_size=3, strides=1, padding='same', theta=0.2, **kwargs):
        super(CentralDifferenceConv2D, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.theta = theta
        self.conv = Conv2D(num_filters, kernel_size, strides=strides, padding=padding)

    def build(self, input_shape):
        super(CentralDifferenceConv2D, self).build(input_shape)

    def call(self, inputs):
        out_normal = self.conv(inputs)  # Standard convolution

        def apply_cdc():
            kernel_diff = K.sum(self.conv.kernel, axis=[0, 1], keepdims=True)
            out_diff = K.conv2d(inputs, kernel_diff, strides=[self.strides, self.strides], padding=self.padding)
            return out_normal - self.theta * out_diff

        def no_cdc():
            return out_normal

        return K.switch(K.abs(self.theta) < 1e-8, no_cdc(), apply_cdc())

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

def channel_attention(input_tensor, reduction_ratio=16):
    channels = int(input_tensor.shape[-1])

    # Global Average Pooling
    pooled_tensor = GlobalAveragePooling2D()(input_tensor)
    pooled_tensor = Reshape((1, 1, channels))(pooled_tensor)

    # Two fully connected layers forming a bottleneck
    reduced_channels = channels // reduction_ratio
    dense_1 = Dense(reduced_channels, activation='relu')(pooled_tensor)
    dense_2 = Dense(channels, activation='sigmoid')(dense_1)

    # Apply channel attention weights
    output_tensor = Multiply()([input_tensor, dense_2])

    return output_tensor

def cal_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    params = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('flops'+ str(flops.total_float_ops))
    print('params'+ str(params.total_parameters))

def generate_Q(H, W):
    coords = np.array([(i, j) for i in range(H) for j in range(W)])
    N = H * W
    coords = coords.reshape(N, 1, 2)
    diffs = coords - coords.transpose(1, 0, 2)
    dists = np.linalg.norm(diffs, axis=2)
    return dists

def FCD(Input):
    x = Input

    # Existing convolutional layers
    x1 = Conv2D(filters=64, kernel_size=11, activation='relu', strides=2, padding='same')(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(filters=64, kernel_size=7, activation='relu', strides=2, padding='same')(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x3 = BatchNormalization()(x3)

    x4 = CentralDifferenceConv2D(num_filters=64, kernel_size=3, strides=2, padding='same')(x)
    x4 = BatchNormalization()(x4)

    x5 = CentralDifferenceConv2D(num_filters=64, kernel_size=3, strides=2, padding='same')(x)
    x5 = BatchNormalization()(x5)

    x6 = CentralDifferenceConv2D(num_filters=64, kernel_size=3, strides=2, padding='same')(x)
    x6 = BatchNormalization()(x6)

    # Concatenate the outputs
    ms = concatenate([x1, x2, x3, x4, x5, x6])

    # Now reshape ms into (batch_size, N, C)
    # N = H * W
    # Get H and W from ms.shape
    shape_ms = K.int_shape(ms)
    H = shape_ms[1]
    W = shape_ms[2]
    C = shape_ms[3]
    N = H * W

    # Reshape ms
    ms_reshaped = Reshape((N, C))(ms)

    # Generate the neighbor indices matrix q_mat_layer
    num_neighbors = 3
    dists = generate_Q(H, W)
    q_mat_layer = np.argsort(dists, axis=1)[:, 1:num_neighbors+1]  # Exclude self (distance zero)

    # Apply GraphConv
    gcn1 = GraphConv(filters=64, neighbors_ix_mat=q_mat_layer, num_neighbors=3, activation='elu')(ms_reshaped)
    gcn2 = GraphConv(filters=64, neighbors_ix_mat=q_mat_layer, num_neighbors=3, activation='elu')(ms_reshaped)
    G = concatenate([gcn1, gcn2], axis=1)
    # Reshape back to (batch_size, H, W, filters)
    gcn_output_reshaped = Reshape((H, W, 128))(G)

    # Continue with the rest of the model
    x = Conv2D(filters=96, kernel_size=1, activation='relu', strides=1, padding='same')(gcn_output_reshaped)

    return x

def attention_spatial(inputs2):
    a = Dense((inputs2.shape[3]).value, activation='softmax')(inputs2)
    return a

def attention_vertical(inputs):
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1, 2))(inputs)
    a = Reshape((input_dim3, input_dim2, input_dim1))(a)
    a = Dense(input_dim2, activation='softmax')(a)

    a_probs = Permute((3, 2, 1))(a)
    return a_probs

def attention_horizontal(inputs2):
    input_dim1 = int(inputs2.shape[1])
    input_dim2 = int(inputs2.shape[2])
    input_dim3 = int(inputs2.shape[3])

    a = Permute((3, 2, 1))(inputs2)
    a = Reshape((input_dim3, input_dim2, input_dim1))(a)
    a = Dense(input_dim2, activation='softmax')(a)

    b_probs = Permute((3, 2, 1))(a)
    return b_probs

class GGA(Layer):  # GGA：gg—attention

    def __init__(self, units, activation=None, **kwargs):
        super(GGA, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kan_layer = KANSplineLayer(in_features=128, out_features=128)
        
        # 为 GGA 层添加 BatchNormalization
        self.bn_gaze = BatchNormalization()
        self.bn_glance = BatchNormalization()

    def call(self, x, **kwargs):
        training = kwargs.get('training', False)
        assert isinstance(x, list), "Input should be a list"

        dk = tf.cast(self.units, tf.float32)
        dk2 = tf.sqrt(dk)
        X, Q, KK, V = x

        max1 = tf.maximum(Q, KK)
        max2 = tf.maximum(max1, V)
        maxx = max2

        # 计算自注意力
        self_att = Activation('softmax')(Q * KK) * V
        gaze = (Q * KK * V) / tf.exp(-maxx / dk2)

        # 通过 KANTensorFlow 层
        X = self.kan_layer(X, training=training)

        # 连接 X 和 self_att
        glance = Concatenate(axis=3)([X, self_att])#拼接在channel轴上

        # 使用 BatchNormalization 层，传递 training 参数
        gaze = self.bn_gaze(gaze, training=training)
        glance = self.bn_glance(glance, training=training)
        
        return [gaze, glance]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list), "Input shape should be a list"
        # 假设输入 X 的形状为 (batch_size, H, W, units)
        batch_size, H, W, C = input_shape[0]
        return [
            (batch_size, H, W, C),
            (batch_size, H, W, 2 * C)
        ]

    def get_config(self):
        config = super(GGA, self).get_config()
        config.update({
            'units': self.units,
            'activation': activations.serialize(self.activation),
        })
        return config
    
class RE_ope(Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(RE_ope, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, list)
        M1, M2 = x
        n = 1
        reward = n * M1
        punishment = tf.zeros_like(M1)
        M1 = tf.where(M1 > 0.2, x=reward, y=punishment)

        A = tf.multiply(M1, M2)

        return A

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0], image_size, image_size, 1 * self.units)
        return input_dim

def RE_module(xx):
    m1 = Activation('swish')(xx)
    m2 = Activation('tanh')(xx)
    RE = RE_ope(128)([m1, m2])
    x = BatchNormalization()(RE)
    return x

class TDA(Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(TDA, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, list)

        X, A1, A2, A3 = x
        A = (A1 + A2 + A3)
        concatenate2 = tf.multiply(A, X)

        max1 = tf.maximum(A1, A2)

        max = tf.maximum(max1, A3)
        concatenate3 = K.concatenate([X, max], axis=3)
        return [concatenate2, concatenate3]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim1 = (input_shape[0][0], image_size, image_size, 1 * self.units)
        input_dim2 = (input_shape[0][0], image_size, image_size, 2 * self.units)
        return [input_dim1, input_dim2]

def demo(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')
    x = FCD(CNNInput)
    x = AveragePooling2D(2, strides=1)(x)
    x1 = BatchNormalization()(x)

    xx = Conv2D(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(x1)
    
    x = RE_module(xx)
    att_3 = attention_spatial(x)
    att_x2 = attention_vertical(x)
    att_x = attention_horizontal(x)
    G1, L2 = GGA(128)([x, att_x, att_x2, att_3])

    L2 = Reshape((81, 256))(L2)
    L2 = Conv1D(filters=384, kernel_size=3, strides=3, activation='relu')(L2)
    x = BatchNormalization()(x)
    L2 = Reshape((9, 9, 128))(L2)
    L2 = BatchNormalization()(L2)
    x = concatenate([G1, L2])
    x = AveragePooling2D(2, strides=1)(x)
    x2 = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=3, activation='relu', strides=1, padding='same')(x2)
    x = BatchNormalization()(x)
    att_3 = attention_spatial(x)
    att_x2 = attention_vertical(x)
    att_x = attention_horizontal(x)
    G1, L2 = GGA(128)([x, att_x, att_x2, att_3])

    L2 = Reshape((64, 256))(L2)
    L2 = Conv1D(filters=512, kernel_size=3, strides=4, activation='relu')(L2)
    x = BatchNormalization()(x)
    L2 = Reshape((8, 8, 128))(L2)
    L2 = BatchNormalization()(L2)
    x = concatenate([G1, L2])
    x = AveragePooling2D(2, strides=1)(x)
    x3 = BatchNormalization()(x)

    x1 = AveragePooling2D(2)(x1)
    x2 = AveragePooling2D(2)(x2)
    x3 = AveragePooling2D(4, strides=1)(x3)

    x = concatenate([x1, x2, x3])
    x = Conv2D(filters=64, kernel_size=3, activation='relu', strides=1)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    x = Dense(256, activation='relu')(x)
    F = Dropout(0.5)(x)

    n = math.ceil(math.sqrt(K.int_shape(F)[-1]))
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=n, c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=[F])
    print(model.summary())
    cal_flops(model)
    return model

def get_model(img_rows, img_cols, num_PC, nb_classes, dataID=1, type='aspn', lr=0.001):
    if num_PC == 0:
        num_PC = image_size_dict[str(dataID)][2]
    if type == 'demo':
        model = demo(img_rows, img_cols, num_PC, nb_classes)
    elif type == 'demo2':
        model = demo2(img_rows, img_cols, num_PC, nb_classes)
    else:
        print('invalid model type, default use demo1 model')
        model = demo1(img_rows, img_cols, num_PC, nb_classes)
    #rivy = RivyOptimizer(lr=lr,rho=0.9, gv_decay=0.00001, momentum=0.9, epsilon=1e-7,growth_factor_coefficient=0.1)
    rmsp = RMSprop(lr=lr, rho=0.9, epsilon=1e-05)
    lhr = LookaheadRMSprop(lr=1e-3, rho=0.9, k=5, alpha=0.5)
    lha = LookaheadAdamW(lr=lr, beta_1=0.9,beta_2=0.999, epsilon=1e-08, decay=1e-3, weight_decay=1e-4, k=5, alpha=0.5)
    model.compile(optimizer=xxx, loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return model

class Symmetry(Initializer):
    """N*N*C Symmetry initial"""
    def __init__(self, n=200, c=16, seed=0):
        self.n = n
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rv = K.truncated_normal([self.n, self.n, self.c], 0., 1e-5, dtype=dtype, seed=self.seed)
        rv = (rv + K.permute_dimensions(rv, pattern=(1, 0, 2))) / 2.0
        return K.reshape(rv, [self.n * self.n, self.c])

def demo1(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=num_PC, c=nb_classes))(CNNInput)
    model = Model(inputs=[CNNInput], outputs=F)
    return model

def get_callbacks(decay=0.0001):
    def step_decay(epoch, lr):
        return lr * math.exp(-1 * epoch * decay)

    callbacks = []
    callbacks.append(LearningRateScheduler(step_decay))

    return callbacks