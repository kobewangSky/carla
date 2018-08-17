
from network import Network


class ICNet_BN(Network):
    def setup(self, is_training, num_classes, evaluation):
# 1
        (self.feed('data')#1024 * 1024 * 3
#1/2
        .interp(s_factor=0.5, name='data_sub2')#512 * 512 * 3
#1/4
        .conv(3,3,32,2,2, biased=False, padding='SAME',relu=False, name='conv1_1_3x3_s2')#256 * 256 * 32
        .batch_normalization(relu=True, name='conv1_1_3x3_s2_bn')
        .conv(3,3,32,1,1, biased=False, padding='SAME', relu=False, name='conv1_2_3x3')#256 * 256 * 32
        .batch_normalization(relu=True, name='conv1_2_3x3_bn')
        .conv(3,3,64,1,1, biased=False, padding='SAME', relu=False, name='conv1_3_3x3')#256 * 256 * 64
        .batch_normalization(relu=True, name='conv1_3_3x3_bn')
        .zero_padding(paddings=1, name='padding0')
#1/8
        .max_pool(3,3,2,2, name='pool1_3x3_s2')#128*128*64
        .conv(1,1,128,1,1, biased=False, relu=False, name='conv2_1_1x1_proj')#128*128*128
        .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

# 3個 Residual Block
        (self.feed('pool1_3x3_s2')
         .conv(1,1,32,1,1, biased=False, relu=False, name='conv2_1_1x1_reduce')#128*128*32
         .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding1')
         .conv(3,3,32,1,1, biased=False, relu=False, name='conv2_1_3x3')#128*128*32
         .batch_normalization(relu=True, name='conv2_1_3x3_bn')
         .conv(1,1,128,1,1, biased=False, relu=False, name='conv2_1_1x1_increase')#128*128*128
         .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn', 'conv2_1_1x1_increase_bn')
         .add(name='conv2_1')
         .relu(name='conv2_1/relu')
         .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')#128*128*32
         .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding2')
         .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_2_3x3')#128*128*32
         .batch_normalization(relu=True, name='conv2_2_3x3_bn')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')#128*128*128
         .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu', 'conv2_2_1x1_increase_bn')
         .add(name='conv2_2')
         .relu(name='conv2_2/relu')
         .conv(1, 1, 32, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')#128*128*32
         .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding3')
         .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv2_3_3x3')#128*128*32
         .batch_normalization(relu=True, name='conv2_3_3x3_bn')
         .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')#128*128*128
         .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))


        (self.feed('conv2_2/relu', 'conv2_3_1x1_increase_bn')
         .add(name='conv2_3')
         .relu(name='conv2_3/relu')
#1/16
         .conv(1,1,256,2,2,biased=True, relu=False, name='conv3_1_1x1_proj')#64*64*256
         .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

# 1個 Residual Block
        (self.feed('conv2_3/relu')
         .conv(1, 1, 64, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')#64*64*64
         .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding4')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv3_1_3x3')#64*64*64
         .batch_normalization(relu=True, name='conv3_1_3x3_bn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')#64*64*256
         .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))


        (self.feed('conv3_1_1x1_proj_bn', 'conv3_1_1x1_increase_bn')
         .add(name='conv3_1')
         .relu(name='conv3_1/relu')
#1/32
         .interp(s_factor=0.5, name='conv3_1_sub4')#32*32*256
         .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')#32*32*64
         .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
         .zero_padding(paddings=1, name='padding5')
         .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv3_2_3x3')#32*32*64
         .batch_normalization(relu=True, name='conv3_2_3x3_bn')
         .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')#32*32*256
         .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

# 2個 Residual Block
        (self.feed('conv3_1_sub4',
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
             .batch_normalization(relu=True, name='conv3_3_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))


        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
             .batch_normalization(relu=True, name='conv3_4_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

#擴增 256 - 512

        (self.feed('conv3_3/relu','conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))
#Dlated block * 5


        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_3_3x3')
             .batch_normalization(relu=True, name='conv4_3_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_4_3x3')
             .batch_normalization(relu=True, name='conv4_4_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_5_3x3')
             .batch_normalization(relu=True, name='conv4_5_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv4_6_3x3')
             .batch_normalization(relu=True, name='conv4_6_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

#擴增 512 - 1024

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding14')
             .atrous_conv(3, 3, 256, 4, biased=False, relu=False, name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))

#Dlated block * 2

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding15')
             .atrous_conv(3, 3, 256, 4, biased=False, relu=False, name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding16')
             .atrous_conv(3, 3, 256, 4, biased=False, relu=False, name='conv5_3_3x3')
             .batch_normalization(relu=True, name='conv5_3_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))

#PSP

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        shape = self.layers['conv5_3/relu'].get_shape().as_list()[1:3]
        h, w = shape

        (self.feed('conv5_3/relu')
             .avg_pool(h, w, h, w, name='conv5_3_pool1')
             .resize_bilinear(shape, name='conv5_3_pool1_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/2, w/2, h/2, w/2, name='conv5_3_pool2')
             .resize_bilinear(shape, name='conv5_3_pool2_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/3, w/3, h/3, w/3, name='conv5_3_pool3')
             .resize_bilinear(shape, name='conv5_3_pool3_interp'))

        (self.feed('conv5_3/relu')
             .avg_pool(h/4, w/4, h/4, w/4, name='conv5_3_pool6')
             .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .add(name='conv5_3_sum')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv5_4_k1')
             .batch_normalization(relu=True, name='conv5_4_k1_bn')
#1/16
             .interp(z_factor=2.0, name='conv5_4_interp')
             .zero_padding(paddings=2, name='padding17')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv_sub4')
             .batch_normalization(relu=False, name='conv_sub4_bn'))


        (self.feed('conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_1_sub2_proj')
             .batch_normalization(relu=False, name='conv3_1_sub2_proj_bn'))


        (self.feed('conv_sub4_bn',
                   'conv3_1_sub2_proj_bn')
             .add(name='sub24_sum')
             .relu(name='sub24_sum/relu')
#1/8
             .interp(z_factor=2.0, name='sub24_sum_interp')
             .zero_padding(paddings=2, name='padding18')
             .atrous_conv(3, 3, 128, 2, biased=False, relu=False, name='conv_sub2')
             .batch_normalization(relu=False, name='conv_sub2_bn'))

        (self.feed('data')
             .conv(3, 3, 32, 2, 2, biased=False, padding='SAME', relu=False, name='conv1_sub1')
             .batch_normalization(relu=True, name='conv1_sub1_bn')
             .conv(3, 3, 32, 2, 2, biased=False, padding='SAME', relu=False, name='conv2_sub1')
             .batch_normalization(relu=True, name='conv2_sub1_bn')
             .conv(3, 3, 64, 2, 2, biased=False, padding='SAME', relu=False, name='conv3_sub1')
             .batch_normalization(relu=True, name='conv3_sub1_bn')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_sub1_proj')
             .batch_normalization(relu=False, name='conv3_sub1_proj_bn'))

        (self.feed('conv_sub2_bn',
                   'conv3_sub1_proj_bn')
             .add(name='sub12_sum')
             .relu(name='sub12_sum/relu')
             .interp(z_factor=2.0, name='sub12_sum_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='conv6_cls'))

        (self.feed('conv5_4_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='sub4_out'))

        (self.feed('sub24_sum_interp')
             .conv(1, 1, num_classes, 1, 1, biased=True, relu=False, name='sub24_out'))


