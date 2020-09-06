#encoding=utf-8
import paddle.fluid as fluid
from nn import ReflectionPad2d, Tanh, ReLU,LeakyReLU,Upsample,Spectralnorm,InNorm,LaNorm
from paddle.fluid.dygraph.nn import Conv2D, Linear,InstanceNorm
from paddle.fluid.dygraph import Sequential


class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False,light_pool_size=1):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.light_pool_size = light_pool_size
        self.ngf = ngf #每一层的基础通道数
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light
        # Encoder Down-Sampling
        DownBlock = []
        DownBlock += [ReflectionPad2d(3),
                      Conv2D(input_nc, ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      ReLU()]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i #i:0,1-->mult:1,2
            DownBlock += [ReflectionPad2d(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0),
                          InstanceNorm(ngf * mult * 2),
                          ReLU()]
        # Encoder Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            #指定该block的输入输出通道数都为ngf*mult
            DownBlock += [ResnetBlock(ngf * mult)]

        # Class Activation Map
        #CAM of Generator
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        #conv1x1中输入通道数乘以2，是因为输入是concat了gap和gmp两个特征图
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1,act='relu')

        # Gamma, Beta block
        if self.light:
            FC = [Linear(self.light_pool_size*self.light_pool_size*ngf * mult, ngf * mult, bias_attr=False,act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu')]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False,act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu')]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Decoder Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Decoder Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Upsample(scale_factor=2, resample='nearest'),
                         ReflectionPad2d(1),
                         #param_attr=fluid.initializer.MSRAInitializer()
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         ReLU()]

        UpBlock2 += [ReflectionPad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False,act='tanh')]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        #Encoder
        x = self.DownBlock(input) #x对应论文的Encoder feature map
        gap = fluid.layers.adaptive_pool2d(x,[1,1],pool_type="avg")
        gap_logit = self.gap_fc(fluid.layers.reshape(gap, shape=[x.shape[0], -1]))
        gap_weight = fluid.layers.reshape(list(self.gap_fc.parameters())[0],shape=[input.shape[0],-1])
        gap = x * fluid.layers.unsqueeze(gap_weight,axes=[2,3])
        gmp = fluid.layers.adaptive_pool2d(x,[1,1],pool_type="max")
        gmp_logit = self.gmp_fc(fluid.layers.reshape(gmp, shape=[x.shape[0], -1]))
        gmp_weight = fluid.layers.reshape(list(self.gmp_fc.parameters())[0],shape=[input.shape[0],-1])
        gmp = x * fluid.layers.unsqueeze(gmp_weight,axes=[2,3])
        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        #gap和gmp对应论文的w*E
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        #x对应论文a1,a2,...,an
        x = self.conv1x1(x) #通道减半
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        #维度转换
        #light和full的区别就是在送入全连接层前，light多了个全局平均池化，将参数个数降低了w*h倍
        #为了提升light的效果，此处可以不降低这么多，池化窗口小一点
        if self.light:
            x_ = fluid.layers.adaptive_pool2d(x, [self.light_pool_size, self.light_pool_size], pool_type="avg")
            x_ = self.FC(fluid.layers.reshape(x_,shape=[x_.shape[0], -1]))
        else:
            x_ = self.FC(fluid.layers.reshape(x,shape=[x.shape[0], -1]))
        # activation map送入全连接层，得到gamma,beta
        gamma, beta = self.gamma(x_), self.beta(x_)
        #decoder
        for i in range(self.n_blocks):
            #UpBlock1_里面使用了ResnetAdaILNBlock，AdaILN的gamma,beta参数是通过全连接层学习到的
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)
        return out, cam_logit, heatmap


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       ReLU()]

        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias=False):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU()

        self.pad2 = ReflectionPad2d(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        out = out + x
        return out

class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(0.9))
        self.rho.stop_gradient = False
        self.in_norm = InNorm()
        self.ln_norm = LaNorm()

    def forward(self, input, gamma, beta):
        out_in = self.in_norm(input)
        out_ln = self.ln_norm(input)
        self.rho.set_value(fluid.layers.clamp(self.rho, 0, 1))
        out = self.rho* out_in + (1-self.rho) * out_ln

        out = out * fluid.layers.unsqueeze(gamma, axes=[2, 3]) + fluid.layers.unsqueeze(beta, axes=[2, 3])
        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(0.0))
        self.rho.stop_gradient = False
        self.gamma = fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(1.0))
        self.gamma.stop_gradient = False
        self.beta = fluid.layers.create_parameter(
            shape=[1, num_features, 1, 1],
            dtype='float32',
            default_initializer=fluid.initializer.Constant(0.0))
        self.beta.stop_gradient = False
        self.in_norm = InNorm()
        self.ln_norm = LaNorm()


    def forward(self, input):
        out_in = self.in_norm(input)
        out_ln = self.ln_norm(input)
        self.rho.set_value(fluid.layers.clamp(self.rho, 0, 1))
        out = self.rho * out_in + (1-self.rho) * out_ln
        out = out * self.gamma+ self.beta
        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        #Encoder Down-sampling
        model = [ReflectionPad2d(1),
                 Spectralnorm(Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0),dim=1),
                 LeakyReLU(0.2, True)]
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(1),
                      Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0),dim=1),
                      LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)

        model += [ReflectionPad2d(1),
                  Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0),dim=1),
                  LeakyReLU(0.2, True)]

        # Class Activation Map
        #CAM of Discriminator
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False),dim=0)
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False),dim=0)
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1)
        self.leaky_relu = LeakyReLU(0.2, True)

        self.pad = ReflectionPad2d(1)
        self.conv = Spectralnorm(Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False),dim=1)

        self.model = Sequential(*model) #encoder部分

    def forward(self, input):
        x = self.model(input)
        gap = fluid.layers.adaptive_pool2d(x, [1, 1], pool_type="avg")
        gap_logit = self.gap_fc(fluid.layers.reshape(gap, shape=[x.shape[0], -1]))
        gap_weight = fluid.layers.reshape(list(self.gap_fc.parameters())[0], shape=[input.shape[0], -1])
        gap = x * fluid.layers.unsqueeze(gap_weight, axes=[2, 3])
        gmp = fluid.layers.adaptive_pool2d(x, [1, 1], pool_type="max")
        gmp_logit = self.gmp_fc(fluid.layers.reshape(gmp, shape=[x.shape[0], -1]))
        gmp_weight = fluid.layers.reshape(list(self.gmp_fc.parameters())[0],shape=[input.shape[0],-1])
        gmp = x * fluid.layers.unsqueeze(gmp_weight, axes=[2, 3])
        cam_logit = fluid.layers.concat(input=[gap_logit, gmp_logit], axis=1)
        x = fluid.layers.concat(input=[gap, gmp], axis=1)
        x = self.leaky_relu(self.conv1x1(x))
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        x = self.pad(x)
        logit = self.conv(x)

        return logit, cam_logit, heatmap