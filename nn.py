import paddle
import math
from paddle.fluid.dygraph import Layer
from paddle import fluid


class MSELoss():
    def __init__(self):
        pass

    def __call__(self, prediction, label):
        return fluid.layers.mse_loss(prediction, label)

class L1Loss():
    def __init__(self):
        pass
    
    def __call__(self, prediction, label):
        return fluid.layers.reduce_mean(fluid.layers.elementwise_sub(prediction, label, act='abs'))

class ReflectionPad2d(Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")


class LeakyReLU(Layer):
    def __init__(self, alpha, inplace=False):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return fluid.layers.leaky_relu(x, self.alpha)

class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return fluid.layers.relu(x)

class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return fluid.layers.tanh(x)


class Dropout(Layer):
    def __init__(self, prob, mode='upscale_in_train'):
        super(Dropout, self).__init__()
        self.prob = prob
        self.mode = mode

    def forward(self, x):
        return fluid.layers.dropout(x, self.prob, dropout_implementation=self.mode)


class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out

class Upsample(fluid.dygraph.Layer):
    def __init__(self,scale_factor=2,resample="NEAREST"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.resample = resample

    def forward(self, input):
        y = fluid.layers.image_resize(input, scale=self.scale_factor,resample="NEAREST")
        return y


def spectral_norm(input,dim=1):
    ret = fluid.layers.spectral_norm(input.weight,dim)
    input.weight.set_value(ret)
    return input

class Spectralnorm(fluid.dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = fluid.dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype,is_bias=False)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

class InNorm(fluid.dygraph.Layer):
    def __init__(self,eps=1e-5):
        super(InNorm, self).__init__()
        self.eps = eps

    paddle.reader.multiprocess_reader
    def forward(self, input):
        in_mean = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True)
        in_var = fluid.layers.reduce_mean(
            (input - in_mean) ** 2, dim=[2, 3], keep_dim=True)
        # out_in:[N,128,128,128]
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        return out_in

class LaNorm(fluid.dygraph.Layer):
    def __init__(self,eps=1e-5):
        super(LaNorm, self).__init__()
        self.eps = eps

    def forward(self, input):
        ln_mean = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True)
        ln_var = fluid.layers.reduce_mean(
            (input - ln_mean) ** 2, dim=[1, 2, 3], keep_dim=True)
        # out_ln:[N,128,128,128]
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        return out_ln

class Pad2D(fluid.dygraph.Layer):
    def __init__(self, paddings, mode, pad_value=0.0):
        super(Pad2D, self).__init__()
        self.paddings = paddings
        self.mode = mode

    def forward(self, x):
        return fluid.layers.pad2d(x, self.paddings, self.mode)
