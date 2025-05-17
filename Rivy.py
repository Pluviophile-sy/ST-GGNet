from keras.optimizers import Optimizer
import keras.backend as K
import numpy as np


# 设置优化器参数


"""
def Rivy_params():
    return{
    'learning_rate': 0.0001,
    'rho': 0.9,
    'gv_decay': 0.0001,
    'momentum': 0.1,
    'epsilon': 1e-7,
    'growth_factor_coefficient': 0.1
}
"""

class RivyOptimizer(Optimizer):
    def __init__(self, learning_rate=1e-3, rho=0.9, gv_decay=0.00001, momentum=0.9, epsilon=1e-7,
                 growth_factor_coefficient=0.1,  noise_mean=0.0, noise_stddev=1.0,**kwargs):
        super(RivyOptimizer, self).__init__(**kwargs)
        self.lr = K.variable(learning_rate, name='lr') 
        self.rho = rho
        self.gv_decay = gv_decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.growth_factor_coefficient = growth_factor_coefficient
        self.noise_mean = noise_mean
        self.noise_stddev = noise_stddev

    def get_updates(self, loss, params):
        # 获取梯度
        grads = self.get_gradients(loss, params)
        self.updates = []

        # 初始化 slots
        for param in params:
            shape = K.int_shape(param)
            rms = K.zeros(shape)  # RMSprop 的累计量
            momentum = K.zeros(shape)  # 动量
            gv = K.zeros(shape)  # 生长速度

            # 获取当前梯度
            grad = grads[params.index(param)]

            # 更新 RMSprop 的累计量
            new_rms = self.rho * rms + (1 - self.rho) * K.square(grad)

            # 计算扰动因子
            growth_factor = self.growth_factor_coefficient * K.random_normal(shape, mean=self.noise_mean, stddev=self.noise_stddev) * grad

            # 更新生长因子 
            new_gv = self.gv_decay * gv + growth_factor

            # 更新生长速度
            new_momentum = self.momentum * momentum + new_gv

            # 计算参数更新量
            new_param = param - self.lr * grad / (K.sqrt(new_rms) + self.epsilon) - self.lr * new_momentum

            # 更新 slots
            self.updates.append(K.update(rms, new_rms))
            self.updates.append(K.update(gv, new_gv))
            self.updates.append(K.update(momentum, new_momentum))

            # 更新参数
            self.updates.append(K.update(param, new_param))

        return self.updates

    def get_config(self):
        # 返回配置，用于保存和加载优化器
        config = {
            'learning_rate': float(K.get_value(self.lr)),
            'rho': self.rho,
            'gv_decay': self.gv_decay,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'growth_factor_coefficient': self.growth_factor_coefficient
        }
        base_config = super(RivyOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''
#实现动态GV版本
class RivyOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, gv_decay=0.9, momentum=0.0, epsilon=1e-7,
                 growth_factor_coefficient=0.1, max_gv=1.0, **kwargs):
        super(RivyOptimizer, self).__init__(**kwargs)
        self.lr = K.variable(learning_rate, name='lr')
        self.rho = rho
        self.gv_decay = gv_decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.growth_factor_coefficient = growth_factor_coefficient
        self.max_gv = max_gv  # 最大的 GV 值
        self.current_epoch = 0  # 当前 epoch

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        for param in params:
            shape = K.int_shape(param)
            rms = K.zeros(shape)
            momentum = K.zeros(shape)
            gv = K.zeros(shape)

            grad = grads[params.index(param)]

            new_rms = self.rho * rms + (1 - self.rho) * K.square(grad)

            # 动态调整 GV
            growth_factor = self.growth_factor_coefficient * K.random_uniform(shape) * grad
            # 随着 epoch 增加 GV
            dynamic_gv = K.minimum(self.max_gv, growth_factor * (self.current_epoch + 1) / 10)  # 你可以调整这个公式
            new_gv = self.gv_decay * gv + dynamic_gv

            new_momentum = self.momentum * momentum + new_gv

            new_param = param - self.lr * grad / (K.sqrt(new_rms) + self.epsilon) - self.lr * new_momentum

            self.updates.append(K.update(rms, new_rms))
            self.updates.append(K.update(gv, new_gv))
            self.updates.append(K.update(momentum, new_momentum))
            self.updates.append(K.update(param, new_param))

        # 更新当前 epoch
        self.current_epoch += 1

        return self.updates
'''
