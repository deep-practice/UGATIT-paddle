#encoding=utf-8
import transforms
import dataset
from networks import *
from utils import *
import paddle.fluid as fluid
import glob,time
import paddle
from nn import BCEWithLogitsLoss

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res
        self.light_pool_size = args.light_pool_size

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = fluid.CPUPlace() if "cpu" in args.device else fluid.CUDAPlace(0)
        self.resume = args.resume
        #self.logger = LogWriter("./log/train")
        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform_A = [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToCHWImage()
        ]
        train_transform_B = [
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotate(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomCrop(self.img_size),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToCHWImage()
        ]
        test_transform = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToCHWImage()
        ]
        self.trainA_loader = paddle.batch(reader=paddle.reader.shuffle(reader=dataset.custom_reader(os.path.join('dataset', self.dataset, 'trainA'),
                     train_transform_A), buf_size=256), batch_size=self.batch_size,drop_last=False)
        self.trainB_loader = paddle.batch(
            reader=paddle.reader.shuffle(reader=dataset.custom_reader(os.path.join('dataset', self.dataset, 'trainB'),
                     train_transform_B), buf_size=256),batch_size=self.batch_size,drop_last=False)
        self.testA_loader = paddle.batch(reader=paddle.reader.shuffle(dataset.custom_reader(os.path.join('dataset', self.dataset, 'testA'),
                     test_transform), buf_size=256), batch_size=self.batch_size,drop_last=False)
        self.testB_loader = paddle.batch(reader = paddle.reader.shuffle(dataset.custom_reader(os.path.join('dataset', self.dataset, 'testB'),
                     test_transform), buf_size=256), batch_size=self.batch_size,drop_last=False)
        with fluid.dygraph.guard(self.device):
            """ Define Generator, Discriminator """
            self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res,
                                          img_size=self.img_size, light=self.light)
            self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res,
                                          img_size=self.img_size, light=self.light)
            # global discriminator
            self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
            self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
            # local discriminator
            self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=3)
            self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=3)

            """ Define Loss """
            self.L1_loss = fluid.dygraph.L1Loss()
            self.MSE_loss = fluid.dygraph.MSELoss()
            self.BCE_loss = BCEWithLogitsLoss()
            self.l2_reg = fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay)
            """ Trainer """
            boundaries = [self.iteration // 2]
            values = [self.lr, self.lr - (self.lr / (self.iteration // 2))]
            self.learning_rate = self.lr

            if self.weight_decay:
                self.learning_rate = fluid.dygraph.PiecewiseDecay(boundaries=boundaries, values=values,begin=0)
            self.G_optim = fluid.optimizer.AdamOptimizer(
                parameter_list=self.genA2B.parameters()+self.genB2A.parameters(),
                learning_rate=self.learning_rate, beta1=0.5, beta2=0.999, regularization=self.l2_reg)
            self.D_optim = fluid.optimizer.AdamOptimizer(
                parameter_list=self.disGA.parameters()+self.disGB.parameters()+self.disLA.parameters()+self.disLB.parameters(),
                learning_rate=self.learning_rate, beta1=0.5, beta2=0.999, regularization=self.l2_reg)


    def train(self):
        with fluid.dygraph.guard(self.device):
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
            start_iter = 1
            if self.resume:
                model_list = glob.glob(os.path.join(self.result_dir,self.dataset,'*.*'))
                if not len(model_list) == 0:
                    model_list.sort()
                    start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                    self.load(os.path.join(self.result_dir, self.dataset), start_iter)
                    print(" [*] Load SUCCESS")

            # training loop
            print('training start !')
            start_time = time.time()
            for step in range(start_iter, self.iteration + 1):
                real_A_iter = next(self.trainA_loader())
                real_B_iter = next(self.trainB_loader())
                real_A, _ = real_A_iter[0][0], real_A_iter[0][1]
                real_B, _ = real_B_iter[0][0], real_B_iter[0][1]

                real_A = real_A[np.newaxis, :]
                real_B = real_B[np.newaxis, :]
                '''
                论文中总共有4种loss:Adversarial loss,Cycle loss,Identity loss,CAM loss
                其中，生成器用到了全部loss,判别器只用到了Adversarial loss和CAM loss
                '''
                real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)
                #set_requires_grad([self.genA2B, self.genB2A], False)
                # Update D
                self.D_optim.clear_gradients()
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_B2A, _, _ = self.genB2A(real_B)

                real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
                real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
                real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
                real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
                #判别器A的目的是让来自A的图片判断为真，生成的A的图片判断为假
                #区分目标域真假图片，见论文公式(6)
                D_ad_loss_GA = self.MSE_loss(real_GA_logit,
                                             fluid.layers.ones_like(real_GA_logit)) + \
                               self.MSE_loss(fake_GA_logit,
                                             fluid.layers.zeros_like(fake_GA_logit))
                D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit,
                                                 fluid.layers.ones_like(real_GA_cam_logit)) + \
                                   self.MSE_loss(fake_GA_cam_logit,
                                                 fluid.layers.zeros_like(fake_GA_cam_logit))
                D_ad_loss_LA = self.MSE_loss(real_LA_logit,
                                             fluid.layers.ones_like(real_LA_logit)) + \
                               self.MSE_loss(fake_LA_logit,
                                             fluid.layers.zeros_like(fake_LA_logit))
                D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit,
                                                 fluid.layers.ones_like(real_LA_cam_logit)) + \
                                   self.MSE_loss(fake_LA_cam_logit,
                                                fluid.layers.zeros_like(fake_LA_cam_logit))
                # 判别器B的目的是让来自B的图片判断为真，生成的B的图片判断为假
                D_ad_loss_GB = self.MSE_loss(real_GB_logit,
                                             fluid.layers.ones_like(real_GB_logit)) + \
                               self.MSE_loss(fake_GB_logit,
                                             fluid.layers.zeros_like(fake_GB_logit))
                D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit,
                                                 fluid.layers.ones_like(real_GB_cam_logit)) + \
                                   self.MSE_loss(fake_GB_cam_logit,
                                                 fluid.layers.zeros_like(fake_GB_cam_logit))
                D_ad_loss_LB = self.MSE_loss(real_LB_logit,
                                             fluid.layers.ones_like(real_LB_logit)) + \
                               self.MSE_loss(fake_LB_logit,
                                             fluid.layers.zeros_like(fake_LB_logit))
                D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit,
                                                 fluid.layers.ones_like(real_LB_cam_logit)) + \
                                   self.MSE_loss(fake_LB_cam_logit,
                                                 fluid.layers.zeros_like(fake_LB_cam_logit))

                D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
                D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

                Discriminator_loss = D_loss_A + D_loss_B
                Discriminator_loss.backward()
                self.D_optim.minimize(Discriminator_loss)
                #set_requires_grad([self.genA2B, self.genB2A], True)
                # Update G
                self.G_optim.clear_gradients()

                fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
                fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

                fake_A2B2A, _, _ = self.genB2A(fake_A2B)  # 把用A域图片生成的B域图片再转到A域
                fake_B2A2B, _, _ = self.genA2B(fake_B2A)  # 把用B域图片生成的A域图片再转到B域

                fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)  # 用A域图片生成A域图片
                fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)  # 用B域图片生成B域图片

                fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
                fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
                fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
                fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)
                #生成器的作用就是以假乱真
                G_ad_loss_GA = self.MSE_loss(fake_GA_logit,
                                             fluid.layers.ones_like(fake_GA_logit))
                G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit,
                                                 fluid.layers.ones_like(fake_GA_cam_logit))
                G_ad_loss_LA = self.MSE_loss(fake_LA_logit,
                                             fluid.layers.ones_like(fake_LA_logit))
                G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit,
                                                 fluid.layers.ones_like(fake_LA_cam_logit))
                G_ad_loss_GB = self.MSE_loss(fake_GB_logit,
                                             fluid.layers.ones_like(fake_GB_logit))
                G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit,
                                                 fluid.layers.ones_like(fake_GB_cam_logit))
                G_ad_loss_LB = self.MSE_loss(fake_LB_logit,
                                             fluid.layers.ones_like(fake_LB_logit))
                G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit,
                                                 fluid.layers.ones_like(fake_LB_cam_logit))
                #图像重建损失
                G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
                G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)
                #让网络学习到恒等映射，源域和目标域属于同一个域的时候，生成的图片不应该改变它
                G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
                G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

                # tune the feature maps in the decoding part be more like the images from the target domain
                # rather than the source domain.
                #似乎跟论文公式(5)表达的不是一个意思？公式(5)同域最大化，不同域最小化
                G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit,
                                             fluid.layers.ones_like(fake_B2A_cam_logit))\
                               + self.BCE_loss(fake_A2A_cam_logit,
                                               fluid.layers.zeros_like(fake_A2A_cam_logit))
                G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit,fluid.layers.ones_like(fake_A2B_cam_logit)) \
                               + self.BCE_loss(fake_B2B_cam_logit,fluid.layers.zeros_like(fake_B2B_cam_logit))

                G_loss_A = self.adv_weight * (
                            G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
                G_loss_B = self.adv_weight * (
                            G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

                Generator_loss = G_loss_A + G_loss_B
                Generator_loss.backward()
                self.G_optim.minimize(Generator_loss)


                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
                step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
                #self.logger.add_scalar(tag="d_loss",step=step,value=Discriminator_loss.numpy())
                #self.logger.add_scalar(tag="g_loss", step=step, value=Generator_loss.numpy())

                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 7, 0, 3))
                    B2A = np.zeros((self.img_size * 7, 0, 3))

                    self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                    for _ in range(train_sample_num):
                        real_A_iter = next(self.trainA_loader())
                        real_B_iter = next(self.trainB_loader())
                        real_A, _ = real_A_iter[0][0], real_A_iter[0][1]
                        real_B, _ = real_B_iter[0][0], real_B_iter[0][1]
                        real_A = real_A[np.newaxis, :]
                        real_B = real_B[np.newaxis, :]
                        real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(denorm(real_A[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_A2A_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_A2A[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_A2B_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_A2B[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_A2B2A_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_A2B2A[0].numpy().transpose(1,2,0)))), 0)),
                                             1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(denorm(real_B[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_B2B_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_B2B[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_B2A_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_B2A[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_B2A2B_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_B2A2B[0].numpy().transpose(1,2,0)))), 0)),
                                             1)
                    for t in range(test_sample_num):
                        real_A_iter_t = next(self.testA_loader())
                        real_B_iter_t = next(self.testB_loader())
                        real_A, _ = real_A_iter_t[0][0], real_A_iter_t[0][1]
                        real_B, _ = real_B_iter_t[0][0], real_B_iter_t[0][1]
                        real_A = real_A[np.newaxis, :]
                        real_B = real_B[np.newaxis, :]
                        real_A, real_B = fluid.dygraph.to_variable(real_A), fluid.dygraph.to_variable(real_B)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)
                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(denorm(real_A[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_A2A_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_A2A[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_A2B_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_A2B[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_A2B2A_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_A2B2A[0].numpy().transpose(1,2,0)))), 0)),
                                             1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(denorm(real_B[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_B2B_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_B2B[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_B2A_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_B2A[0].numpy().transpose(1,2,0))),
                                                                   cam(fake_B2A2B_heatmap[0].numpy().transpose(1,2,0),
                                                                       self.img_size),
                                                                   RGB2BGR(denorm(fake_B2A2B[0].numpy().transpose(1,2,0)))), 0)),
                                             1)

                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                    self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

                if step % self.save_freq == 0:
                    self.save(os.path.join(self.result_dir, self.dataset), step)

                if step % 1000 == 0:
                    fluid.save_dygraph(self.G_optim.state_dict(),os.path.join(self.result_dir, self.dataset,'model','GOptim_latest'))
                    fluid.save_dygraph(self.D_optim.state_dict(),os.path.join(self.result_dir, self.dataset, 'model', 'DOptim_latest'))
                    fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(self.result_dir, self.dataset,'model','genA2B_latest'))
                    fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(self.result_dir, self.dataset,'model','genB2A_latest'))
                    fluid.save_dygraph(self.disGA.state_dict(), os.path.join(self.result_dir, self.dataset ,'model','disGA_latest'))
                    fluid.save_dygraph(self.disGB.state_dict(), os.path.join(self.result_dir, self.dataset ,'model', 'disGB_latest'))
                    fluid.save_dygraph(self.disLA.state_dict(), os.path.join(self.result_dir, self.dataset ,'model', 'disLA_latest'))
                    fluid.save_dygraph(self.disLB.state_dict(), os.path.join(self.result_dir, self.dataset ,'model', 'disLB_latest'))

    def save(self, res_dir, step):
        fluid.save_dygraph(self.G_optim.state_dict(),os.path.join(res_dir, 'GOptim_%07d' % step))
        fluid.save_dygraph(self.D_optim.state_dict(),os.path.join(res_dir, 'DOptim_%07d' % step))
        fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(res_dir,'genA2B_%07d' % step))
        fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(res_dir,'genB2A_%07d' % step))
        fluid.save_dygraph(self.disGA.state_dict(), os.path.join(res_dir,'disGA_%07d' % step))
        fluid.save_dygraph(self.disGB.state_dict(), os.path.join(res_dir,'disGB_%07d' % step))
        fluid.save_dygraph(self.disLA.state_dict(), os.path.join(res_dir,'disLA_%07d' % step))
        fluid.save_dygraph(self.disLB.state_dict(), os.path.join(res_dir, 'disLB_%07d' % step))

    def load(self, res_dir, step):
        genA2B_param,_ = fluid.load_dygraph(os.path.join(res_dir,'genA2B_%07d' % step))
        for k,v in genA2B_param.items():
            if "FC.2.weight" in k:
                genA2B_param["FC.1.weight"] = genA2B_param.pop("FC.2.weight")
                break
        self.genA2B.set_dict(genA2B_param)
        genB2A_param,_ = fluid.load_dygraph(os.path.join(res_dir,'genB2A_%07d' % step))
        for k, v in genB2A_param.items():
            if "FC.2.weight" in k:
                genB2A_param["FC.1.weight"] = genB2A_param.pop("FC.2.weight")
                break
        self.genB2A.set_dict(genB2A_param)
        disGA_param,_ = fluid.load_dygraph(os.path.join(res_dir,'disGA_%07d' % step))
        self.disGA.set_dict(disGA_param)
        disGB_param,_ = fluid.load_dygraph(os.path.join(res_dir,'disGB_%07d' % step))
        self.disGB.set_dict(disGB_param)
        disLA_param,_ = fluid.load_dygraph(os.path.join(res_dir,'disLA_%07d' % step))
        self.disLA.set_dict(disLA_param)
        disLB_param,_ = fluid.load_dygraph(os.path.join(res_dir, 'disLB_%07d' % step))
        self.disLB.set_dict(disLB_param)

    def load_latest(self, res_dir):
        _,gopt_param = fluid.load_dygraph(os.path.join(res_dir,'GOptim_latest'))
        self.G_optim.set_dict(gopt_param)
        _, dopt_param = fluid.load_dygraph(os.path.join(res_dir, 'DOptim_latest'))
        self.D_optim.set_dict(dopt_param)
        genA2B_param,_ = fluid.load_dygraph(os.path.join(res_dir,'genA2B_latest'))
        for k,v in genA2B_param.items():
            if "FC.2.weight" in k:
                genA2B_param["FC.1.weight"] = genA2B_param.pop("FC.2.weight")
                break
        self.genA2B.set_dict(genA2B_param)
        genB2A_param,_ = fluid.load_dygraph(os.path.join(res_dir,'genB2A_latest'))
        for k,v in genB2A_param.items():
            if "FC.2.weight" in k:
                genB2A_param["FC.1.weight"] = genB2A_param.pop("FC.2.weight")
                break
        self.genB2A.set_dict(genB2A_param)
        disGA_param,_ = fluid.load_dygraph(os.path.join(res_dir,'disGA_latest'))
        self.disGA.set_dict(disGA_param)
        disGB_param,_ = fluid.load_dygraph(os.path.join(res_dir,'disGB_latest'))
        self.disGB.set_dict(disGB_param)
        disLA_param,_ = fluid.load_dygraph(os.path.join(res_dir,'disLA_latest'))
        self.disLA.set_dict(disLA_param)
        disLB_param,_ = fluid.load_dygraph(os.path.join(res_dir, 'disLB_latest'))
        self.disLB.set_dict(disLB_param)


    def test(self):
        with fluid.dygraph.guard(self.device):
            model_list = glob.glob(os.path.join(self.result_dir, self.dataset, '*.*'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset), start_iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            self.genA2B.eval(), self.genB2A.eval()
            step = 0
            if not os.path.exists(os.path.join(self.result_dir, self.dataset, 'fakeA2B')):
                os.mkdir(os.path.join(self.result_dir, self.dataset, 'fakeA2B'))
            if not os.path.exists(os.path.join(self.result_dir, self.dataset, 'fakeB2A')):
                os.mkdir(os.path.join(self.result_dir, self.dataset, 'fakeB2A'))

            for data in self.testA_loader():
                real_A, _ = data[0][0], data[0][1]
                real_A = real_A[np.newaxis, :]

                real_A = fluid.dygraph.to_variable(real_A)
                fake_A2B, _, _ = self.genA2B(real_A)
                fake_A2B = RGB2BGR(denorm(fake_A2B[0].numpy().transpose(1,2,0)))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeA2B', 'A2B_%07d.png' % step), fake_A2B*255)
                step+=1
            step = 0
            for data in self.testB_loader():
                real_B, _ = data[0][0], data[0][1]
                real_B = real_B[np.newaxis, :]
                real_B = fluid.dygraph.to_variable(real_B)
                fake_B2A, _, _ = self.genB2A(real_B)
                fake_B2A = RGB2BGR(denorm(fake_B2A[0].numpy().transpose(1, 2, 0)))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeB2A', 'B2A_%07d.png' % step), fake_B2A*255)
                step += 1




