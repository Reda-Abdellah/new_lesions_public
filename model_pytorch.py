import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
########################################

class Generator(nn.Module): #mask_generator
    #this is the implementation of assemblynet architecture from tenserflow/keras model.
    def __init__(self,nf=24,nc=1,dropout_rate=0.5,in_shape=1):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(in_shape, nf, 3, padding= 1)
        self.conv1_bn = nn.BatchNorm3d(nf)
        self.conv2 = nn.Conv3d(nf, 2*nf, 3, padding= 1)
        self.conv2_bn = nn.BatchNorm3d(2*nf)
        self.conv3 = nn.Conv3d(2*nf, 2*nf, 3, padding= 1)
        self.conv3_bn = nn.BatchNorm3d(2*nf)
        self.conv4 = nn.Conv3d(2*nf, 4*nf, 3, padding= 1)
        self.conv4_bn = nn.BatchNorm3d(4*nf)
        self.conv5 = nn.Conv3d(4*nf, 4*nf, 3, padding= 1)
        self.conv5_bn = nn.BatchNorm3d(4*nf)
        self.conv6 = nn.Conv3d(4*nf, 8*nf, 3, padding= 1)
        self.conv6_bn = nn.BatchNorm3d(8*nf)
        self.conv7 = nn.Conv3d(8*nf, 16*nf, 3, padding= 1)
        #bottleneck
        self.concat1_bn = nn.BatchNorm3d(8*nf+16*nf)
        self.conv8 = nn.Conv3d((8*nf+16*nf), 8*nf, 3, padding= 1)

        self.concat2_bn = nn.BatchNorm3d(8*nf+4*nf)
        self.conv9 = nn.Conv3d(8*nf+4*nf, 4*nf, 3, padding= 1)

        self.concat3_bn = nn.BatchNorm3d(4*nf+2*nf)
        self.conv10 = nn.Conv3d(4*nf+2*nf, 4*nf, 3, padding= 1)

        self.conv_out = nn.Conv3d(4*nf, nc, 3, padding= 1)
        self.up= nn.Upsample(scale_factor=2,mode='trilinear', align_corners=False)
        #self.up= nn.Upsample(scale_factor=2,mode='nearest')
        self.pool= nn.MaxPool3d(2)
        self.dropout= nn.Dropout(dropout_rate)
        self.final_activation=nn.Sigmoid()

    def encoder(self,in_x):
        self.x1=self.conv1_bn(F.relu(self.conv1(in_x)))
        self.x1= F.relu(self.conv2(self.x1))
        self.x2= self.conv2_bn(self.dropout(self.pool(self.x1)))
        self.x2=self.conv3_bn(F.relu(self.conv3(self.x2)))
        self.x2= F.relu(self.conv4(self.x2))
        self.x3= self.conv4_bn(self.dropout(self.pool(self.x2)))
        self.x3=self.conv5_bn(F.relu(self.conv5(self.x3)))
        self.x3= F.relu(self.conv6(self.x3))
        self.x4= self.conv6_bn(self.dropout(self.pool(self.x3)))
        self.x4=F.relu(self.conv7(self.x4))#bottleneck

    def decoder(self):
        self.x5=self.up(self.x4)
        self.x5=self.concat1_bn(  torch.cat((self.x5,self.x3), dim=1)  )
        self.x5= F.relu(self.conv8(self.x5))
        self.x6=self.up(self.x5)
        self.x6=self.concat2_bn(  torch.cat((self.x6,self.x2), dim=1)  )
        self.x6= F.relu(self.conv9(self.x6))
        self.x7=self.up(self.x6)
        self.x7=self.concat3_bn(  torch.cat((self.x7,self.x1), dim=1)  )
        self.x7= F.relu(self.conv10(self.x7))
        return self.x7

    def forward(self, x):
        self.encoder(x)
        decoder_out=self.decoder()
        out= self.final_activation(self.conv_out(decoder_out))
        return out

class unet_assemblynet(nn.Module):
    #this is the implementation of assemblynet architecture from tenserflow/keras model.
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv3d(in_mod, nf, 3, padding= 1)
        self.conv1_bn = nn.BatchNorm3d(nf)
        self.conv2 = nn.Conv3d(nf, 2*nf, 3, padding= 1)
        self.conv2_bn = nn.BatchNorm3d(2*nf)
        self.conv3 = nn.Conv3d(2*nf, 2*nf, 3, padding= 1)
        self.conv3_bn = nn.BatchNorm3d(2*nf)
        self.conv4 = nn.Conv3d(2*nf, 4*nf, 3, padding= 1)
        self.conv4_bn = nn.BatchNorm3d(4*nf)
        self.conv5 = nn.Conv3d(4*nf, 4*nf, 3, padding= 1)
        self.conv5_bn = nn.BatchNorm3d(4*nf)
        self.conv6 = nn.Conv3d(4*nf, 8*nf, 3, padding= 1)
        self.conv6_bn = nn.BatchNorm3d(8*nf)
        self.conv7 = nn.Conv3d(8*nf, 16*nf, 3, padding= 1)
        #bottleneck
        self.concat1_bn = nn.BatchNorm3d(8*nf+16*nf)
        #print('hak swalhak')
        self.conv8 = nn.Conv3d((8*nf+16*nf), 8*nf, 3, padding= 1)

        self.concat2_bn = nn.BatchNorm3d(8*nf+4*nf)
        self.conv9 = nn.Conv3d(8*nf+4*nf, 4*nf, 3, padding= 1)

        self.concat3_bn = nn.BatchNorm3d(4*nf+2*nf)
        self.conv10 = nn.Conv3d(4*nf+2*nf, 4*nf, 3, padding= 1)

        self.conv_out = nn.Conv3d(4*nf, nc, 3, padding= 1)
        self.up= nn.Upsample(scale_factor=2,mode='trilinear', align_corners=False)
        #self.up= nn.Upsample(scale_factor=2,mode='nearest')
        self.pool= nn.MaxPool3d(2)
        self.dropout= nn.Dropout(dropout_rate)
        self.softmax=nn.Softmax(dim=1)

    def encoder(self,in_x):
        self.x1=self.conv1_bn(F.relu(self.conv1(in_x)))
        self.x1= F.relu(self.conv2(self.x1))
        self.x2= self.conv2_bn(self.dropout(self.pool(self.x1)))
        self.x2=self.conv3_bn(F.relu(self.conv3(self.x2)))
        self.x2= F.relu(self.conv4(self.x2))
        self.x3= self.conv4_bn(self.dropout(self.pool(self.x2)))
        self.x3=self.conv5_bn(F.relu(self.conv5(self.x3)))
        self.x3= F.relu(self.conv6(self.x3))
        self.x4= self.conv6_bn(self.dropout(self.pool(self.x3)))
        self.x4=F.relu(self.conv7(self.x4))#bottleneck
        return self.x4,self.x3,self.x2,self.x1

    def decoder(self,x4,x3,x2,x1):
        self.x5=self.up(x4)
        self.x5=self.concat1_bn(  torch.cat((self.x5,x3), dim=1)  )
        #self.x5=self.cat1_bn(  torch.cat((self.x5,x3), dim=1)  )
        self.x5= F.relu(self.conv8(self.x5))
        self.x6=self.up(self.x5)
        self.x6=self.concat2_bn(  torch.cat((self.x6,x2), dim=1)  )
        #self.x6=self.cat2_bn(  torch.cat((self.x6,x2), dim=1)  )
        self.x6= F.relu(self.conv9(self.x6))
        self.x7=self.up(self.x6)
        self.x7=self.concat3_bn(  torch.cat((self.x7,x1), dim=1)  )
        #self.x7=self.cat3_bn(  torch.cat((self.x7,x1), dim=1)  )
        self.x7= F.relu(self.conv10(self.x7))
        return self.softmax(self.conv_out(self.x7))

    def forward(self, x):
        x4,x3,x2,x1 = self.encoder(x)
        decoder_out=self.decoder(x4,x3,x2,x1)
        return decoder_out

class unet_assemblynet_groupnorm(unet_assemblynet):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1, group_number=3):
        super().__init__(nf,nc,dropout_rate,in_mod)
        self.conv1_bn = nn.GroupNorm(group_number, nf)
        self.conv2_bn = nn.GroupNorm(2*group_number, 2*nf)
        self.conv3_bn = nn.GroupNorm(2*group_number, 2*nf)
        self.conv4_bn = nn.GroupNorm(4*group_number, 4*nf)
        self.conv5_bn = nn.GroupNorm(4*group_number, 4*nf)
        self.conv6_bn = nn.GroupNorm(8*group_number, 8*nf)
        #bottleneck
        self.concat1_bn = nn.GroupNorm(24*group_number, 24*nf)
        self.concat2_bn = nn.GroupNorm(12*group_number, 12*nf)
        self.concat3_bn = nn.GroupNorm(6*group_number, 6*nf)



class unet_siamese(unet_assemblynet_groupnorm):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1, group_number=3):
        super().__init__(nf,nc,dropout_rate,in_mod)

    def forward(self, x):
        x4_1,x3_1,x2_1,x1_1 = self.encoder(x[:,0:1,:,:,:])
        x4_2,x3_2,x2_2,x1_2 = self.encoder(x[:,1:2,:,:,:])
        x3=F.relu(x3_2-x3_1)
        x2=F.relu(x2_2-x2_1)
        x1=F.relu(x1_2-x1_1)
        decoder_out=self.decoder(x4_1,x3,x2,x1)
        return decoder_out

class unet_siamese_abs(unet_assemblynet_groupnorm):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1, group_number=3):
        super().__init__(nf,nc,dropout_rate,in_mod)

    def forward(self, x):
        x4_1,x3_1,x2_1,x1_1 = self.encoder(x[:,0:1,:,:,:])
        x4_2,x3_2,x2_2,x1_2 = self.encoder(x[:,1:2,:,:,:])
        x4=torch.abs(x4_2-x4_1)
        x3=torch.abs(x3_2-x3_1)
        x2=torch.abs(x2_2-x2_1)
        x1=torch.abs(x1_2-x1_1)
        decoder_out=self.decoder(x4_1,x3,x2,x1)
        return decoder_out

class unet_siamese_fractal(unet_assemblynet_groupnorm):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_mod=1, group_number=3):
        super().__init__(nf,nc,dropout_rate,in_mod)
        self.fusion1=CATFusion3D(nf*2,nf*2, nf*2, norm = 'GroupNorm', norm_groups=group_number*2, ftdepth=5)
        self.fusion2=CATFusion3D(nf*4,nf*4, nf*4, norm = 'GroupNorm', norm_groups=group_number*4, ftdepth=5)
        self.fusion3=CATFusion3D(nf*8,nf*8, nf*8, norm = 'GroupNorm', norm_groups=group_number*8, ftdepth=5)
        self.fusion4=CATFusion3D(nf*16,nf*16, nf*16, norm = 'GroupNorm', norm_groups=group_number*16, ftdepth=5)

    def forward(self, x):
        x4_1,x3_1,x2_1,x1_1 = self.encoder(x[:,0:1,:,:,:])
        x4_2,x3_2,x2_2,x1_2 = self.encoder(x[:,1:2,:,:,:])
        x4=self.fusion4(x4_1, x4_2)
        x3=self.fusion3(x3_1, x3_2)
        x2=self.fusion2(x2_1, x2_2)
        x1=self.fusion1(x1_1, x1_2)
        decoder_out=self.decoder(x4,x3,x2,x1)
        return decoder_out

class generator(unet_assemblynet_groupnorm):
    def __init__(self,nf=24,nc=2,dropout_rate=0.5,in_shape=1):
        super().__init__(nf,nc,dropout_rate,in_shape)
    def decoder(self):
        self.x5=self.up(self.x4)
        self.x5=self.concat1_bn(  torch.cat((self.x5,self.x3), dim=1)  )
        self.x5= F.relu(self.conv8(self.x5))
        self.x6=self.up(self.x5)
        self.x6=self.concat2_bn(  torch.cat((self.x6,self.x2), dim=1)  )
        self.x6= F.relu(self.conv9(self.x6))
        self.x7=self.up(self.x6)
        self.x7=self.concat3_bn(  torch.cat((self.x7,self.x1), dim=1)  )
        self.x7= F.relu(self.conv10(self.x7))
        return self.x7
    def forward(self, x):
        self.encoder(x)
        decoder_out=self.decoder()
        out= self.conv_out(decoder_out)
        return out

########################################
########################################

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x

class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # to use hiddenlayer, make padding=1, output_padding=1,
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        # to use hiddenlayer, comment the usage of center_crop
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out

class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class FCDenseNet(nn.Module):
    # copied from https://github.com/bfortuner/pytorch_tiramisu
    # slightly changed to output range from -1 to 1
    def __init__(self, in_channels=2, down_blocks=(4, 4, 4, 4, 4),
                up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
                 growth_rate=12, out_chans_first_conv=48, out_activation='tanh',n_classes=1):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
        if(out_activation=='tanh'):
            self.out_act = nn.Tanh()
        elif(out_activation=='softmax'):
            self.out_act = nn.Softmax(dim=1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.out_act(out)
        # out = self.softmax(out)
        return out

########################################
########################################

class SiamUnet_diff(nn.Module):
    """SiamUnet_diff segmentation network."""

    def __init__(self, input_nbr=1, out_activation='tanh',n_classes=1, n1 = 16, dropout_rate=0.2):
        super(SiamUnet_diff, self).__init__()


        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.input_nbr = input_nbr
        if(out_activation=='tanh'):
            self.out_act = nn.Tanh()
        elif(out_activation=='softmax'):
            self.out_act = nn.Softmax(dim=1)

        self.conv11 = nn.Conv2d(input_nbr, filters[0], kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(filters[0])
        self.do11 = nn.Dropout2d(p=dropout_rate)
        self.conv12 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(filters[0])
        self.do12 = nn.Dropout2d(p=dropout_rate)

        self.conv21 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(filters[1])
        self.do21 = nn.Dropout2d(p=dropout_rate)
        self.conv22 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(filters[1])
        self.do22 = nn.Dropout2d(p=dropout_rate)

        self.conv31 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(filters[2])
        self.do31 = nn.Dropout2d(p=dropout_rate)
        self.conv32 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(filters[2])
        self.do32 = nn.Dropout2d(p=dropout_rate)
        self.conv33 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(filters[2])
        self.do33 = nn.Dropout2d(p=dropout_rate)

        self.conv41 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(filters[3])
        self.do41 = nn.Dropout2d(p=dropout_rate)
        self.conv42 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(filters[3])
        self.do42 = nn.Dropout2d(p=dropout_rate)
        self.conv43 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(filters[3])
        self.do43 = nn.Dropout2d(p=dropout_rate)

        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(filters[3])
        self.do43d = nn.Dropout2d(p=dropout_rate)
        self.conv42d = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(filters[3])
        self.do42d = nn.Dropout2d(p=dropout_rate)
        self.conv41d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(filters[2])
        self.do41d = nn.Dropout2d(p=dropout_rate)

        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(filters[2])
        self.do33d = nn.Dropout2d(p=dropout_rate)
        self.conv32d = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(filters[2])
        self.do32d = nn.Dropout2d(p=dropout_rate)
        self.conv31d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(filters[1])
        self.do31d = nn.Dropout2d(p=dropout_rate)

        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(filters[1])
        self.do22d = nn.Dropout2d(p=dropout_rate)
        self.conv21d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(filters[0])
        self.do21d = nn.Dropout2d(p=dropout_rate)

        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(filters[0])
        self.do12d = nn.Dropout2d(p=dropout_rate)
        self.conv11d = nn.ConvTranspose2d(filters[0], n_classes, kernel_size=3, padding=1)


    def forward(self, x):
        x1, x2= x[:,0:1,:,:],x[:,1:2,:,:]

        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)



        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        #diff_43=torch.abs(x43_1 - x43_2)
        diff_43=x43_2-x43_1
        x4d = torch.cat((pad4(x4d),diff_43 ), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        #diff_33=torch.abs(x33_1 - x33_2)
        diff_33=x33_2-x33_1
        x3d = torch.cat((pad3(x3d), diff_33 ), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        #diff_22=torch.abs(x22_1 - x22_2)
        diff_22=x22_2-x22_1
        x2d = torch.cat((pad2(x2d),diff_22 ), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        #diff_12=torch.abs(x12_1 - x12_2)
        diff_12=x12_2-x12_1
        x1d = torch.cat((pad1(x1d), diff_12), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)
        #return (x11d, )
        return self.out_act(x11d)

class FCSiamDiff(nn.Module):
    def __init__(self, in_dim=1,out_dim=1, out_activation='tanh'):
        super(FCSiamDiff, self).__init__()
        if(out_activation=='tanh'):
            self.out_act = nn.Tanh()
        elif(out_activation=='softmax'):
            self.out_act = nn.Softmax(dim=1)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0)
        )

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def forward(self, x):
        pre_data, post_data= x[:,0:1,:,:],x[:,1:2,:,:]
        #####################
        # decoder
        #####################
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)

        up_feature_5 = self.up_sample_1(down_feature_41)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)

        return self.out_act(output_feature)

class FCSiamDiff_withSFA(FCSiamDiff):
    def __init__(self, in_dim=1,out_dim=1, out_activation='tanh',batch_size=32):
        super().__init__(in_dim,out_dim, out_activation)
        self.reg = 1e-4
        self.num = batch_size

    def DSFA(self, X, Y):
        X_hat = X - torch.mean(X, axis=0)
        Y_hat = Y - torch.mean(Y, axis=0)

        differ = X_hat - Y_hat
        #print(differ.T.size())
        #print(differ.size())
        A = torch.matmul(differ.permute(0,2,1), differ)
        #A = A/self.num
        A =A.mean(0)
        Sigma_XX = torch.matmul(X_hat.permute(0,2,1), X_hat)
        Sigma_XX = Sigma_XX / self.num + self.reg * torch.eye(Sigma_XX.size(1)).to(device)
        Sigma_YY = torch.matmul(Y_hat.permute(0,2,1), Y_hat)
        Sigma_YY = Sigma_YY / self.num + self.reg * torch.eye(Sigma_XX.size(1)).to(device)

        B = (Sigma_XX+Sigma_YY)/2

        # For numerical stability.
        B=B.mean(0)
        #print(B.size())
        D_B, V_B = torch.eig(B)
        idx = torch.where(D_B > 1e-12)[0]#[:, 0]
        print(D_B.size())
        print(V_B.size())
        D_B = torch.gather(D_B, 0, idx)
        V_B = torch.gather(V_B, 0, idx)
        B_inv = torch.matmul(torch.matmul(V_B, torch.diag(torch.reciprocal(D_B))), V_B.permute(0,2,1))
        ##
        Sigma = torch.matmul(B_inv, A)
        loss = torch.trace(torch.matmul(Sigma, Sigma))
        return loss

    def forward(self, x):
        pre_data, post_data= x[:,0:1,:,:],x[:,1:2,:,:]
        #####################
        # decoder
        #####################
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)

        sizf1=feature_11.size()
        sizf2=feature_21.size()
        sizf3=feature_31.size()
        sizf4=feature_41.size()

        reg_loss_f1= self.DSFA(feature_11.view(sizf1[0],sizf1[1], -1),feature_12.view(sizf1[0],sizf1[1], -1))
        reg_loss_f2= self.DSFA(feature_21.view(sizf2[0],sizf2[1], -1),feature_22.view(sizf2[0],sizf2[1], -1))
        reg_loss_f3= self.DSFA(feature_31.view(sizf3[0],sizf3[1], -1),feature_32.view(sizf3[0],sizf3[1], -1))
        reg_loss_f4= self.DSFA(feature_41.view(sizf4[0],sizf4[1], -1),feature_42.view(sizf4[0],sizf4[1], -1))
        reg_loss= 0.55*reg_loss_f1 + 0.2*reg_loss_f2 + 0.15* reg_loss_f3 + 0.1* reg_loss_f4


        up_feature_5 = self.up_sample_1(down_feature_41)
        concat_feature_5 = torch.cat([up_feature_5, torch.abs(feature_41 - feature_42)], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, torch.abs(feature_31 - feature_32)], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, torch.abs(feature_21 - feature_22)], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, torch.abs(feature_11 - feature_12)], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        out= self.out_act(output_feature)
        return out,reg_loss

class FCSiamConc(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, out_activation='tanh'):
        super(FCSiamConc, self).__init__()
        if(out_activation=='tanh'):
            self.out_act = nn.Tanh()
        elif(out_activation=='softmax'):
            self.out_act = nn.Softmax(dim=1)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0),
        )

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def forward(self, x):
        pre_data, post_data= x[:,0:1,:,:],x[:,1:2,:,:]
        #####################
        # decoder
        #####################
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(pre_data)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(post_data)

        up_feature_5 = self.up_sample_1(down_feature_41)
        concat_feature_5 = torch.cat([up_feature_5, feature_41, feature_42], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, feature_31, feature_32], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, feature_21, feature_22], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, feature_11, feature_12], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        #output = F.softmax(output_feature, dim=1)
        #return output_feature, output
        return self.out_act(output_feature)

class FCEF(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, out_activation='tanh'):
        super(FCEF, self).__init__()
        if(out_activation=='tanh'):
            self.out_act = nn.Tanh()
        elif(out_activation=='softmax'):
            self.out_act = nn.Softmax(dim=1)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1, padding=0),
        )

    def forward(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        #####################
        # decoder
        #####################
        up_feature_5 = self.up_sample_1(down_feature_4)
        concat_feature_5 = torch.cat([up_feature_5, feature_4], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, feature_3], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, feature_2], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, feature_1], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        return self.out_act(output_feature)

########################################
########################################

def get_norm(name, axis=1, norm_groups=None):
    if (name == 'BatchNorm'):
        return nn.BatchNorm2d(axis=axis)
    elif (name == 'InstanceNorm'):
        return nn.InstanceNorm(axis=axis)
    elif (name == 'LayerNorm'):
        return nn.LayerNorm(axis=axis)
    elif (name == 'GroupNorm' and norm_groups is not None):
        return nn.GroupNorm(norm_groups, norm_groups*8) # applied to channel axis
    else:
        raise NotImplementedError

class Conv2DNormed(nn.Module):
    """
        Convenience wrapper layer for 2D convolution followed by a normalization layer
        All other keywords are the same as torch.nn.Conv2d
    """

    def __init__(self,  in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1),   weight_initializer=None,
                    _norm_type = 'BatchNorm', norm_groups=None, axis =1 , groups=1,**kwards):
        super().__init__(**kwards)

        self.conv = nn.Conv2d( in_channels, out_channels, kernel_size = kernel_size,
                                      stride= stride, padding=padding,
                                      dilation= dilation, bias=False, groups=groups)

        self.norm_layer = get_norm(_norm_type, axis=axis, norm_groups= norm_groups)

    def forward(self,_x):
        x = self.conv(_x)
        x = self.norm_layer(x)
        return x

class Conv3DNormed(Conv2DNormed):
    def __init__(self,   in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1,    weight_initializer=None,   _norm_type = 'BatchNorm', norm_groups=None, axis =1 , groups=1,**kwards):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                    padding, dilation, weight_initializer,
                    _norm_type, norm_groups, axis , groups,**kwards)
        self.conv = nn.Conv3d( in_channels, out_channels, kernel_size = kernel_size,
                                      stride= stride, padding=padding,
                                      dilation= dilation, bias=False, groups=groups)

class FTanimoto(nn.Module):
    """
    This is the average fractal Tanimoto set similarity with complement.
    """
    def __init__(self, depth=5, smooth=1.0e-5, axis=[2,3],**kwards):
        super().__init__(**kwards)

        assert depth >= 0, "Expecting depth >= 0, aborting ..."

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1./depth

        self.smooth = smooth
        self.axis=axis

    def inner_prod(self, prob, label):
        prod = torch.mul(prob,label)
        prod = torch.sum(prod,dim=self.axis,keepdims=True)
        return prod

    def tnmt_base(self, preds, labels):

        tpl  = self.inner_prod(preds,labels)
        tpp  = self.inner_prod(preds,preds)
        tll  = self.inner_prod(labels,labels)
        num = tpl + self.smooth
        denum = 0.0
        for d in range(self.depth):
            a = 2.**d
            b = -(2.*a-1.)
            denum = denum + torch.reciprocal(torch.add(a*(tpp+tll), b *tpl) + self.smooth)
        return torch.mul(num,denum)*self.scale

    def forward(self, preds, labels):
            l12 = self.tnmt_base(preds,labels)
            l12 = l12 + self.tnmt_base(1.-preds, 1.-labels)

            return 0.5*l12

class RelFTAttention2D(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None,ftdepth=5,**kwards):
        super().__init__(**kwards)
        self.query  = Conv2DNormed( in_channels, out_channels,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)
        self.key    = Conv2DNormed( in_channels, out_channels,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)
        self.value  = Conv2DNormed( in_channels, out_channels,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)
        self.metric_channel = FTanimoto(depth=ftdepth, axis=[2,3])
        self.metric_space = FTanimoto(depth=ftdepth, axis=1)
        self.norm = get_norm(name=norm, axis=1, norm_groups= norm_groups)

    def forward(self, input1, input2, input3):
        # These should work with ReLU as well
        q = F.sigmoid(self.query(input1))
        k = F.sigmoid(self.key(input2))# B,C,H,W
        v = F.sigmoid(self.value(input3)) # B,C,H,W
        att_spat =  self.metric_space(q,k) # B,1,H,W
        v_spat  =  torch.mul(att_spat, v) # emphasize spatial features
        att_chan =  self.metric_channel(q,k) # B,C,1,1
        v_chan   =  torch.mul(att_chan, v) # emphasize spatial features
        v_cspat =   0.5*torch.add(v_chan, v_spat) # emphasize spatial features
        v_cspat = self.norm(v_cspat)
        return v_cspat

class RelFTAttention3D(RelFTAttention2D):
    def __init__(self,  in_channels, out_channels, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None,ftdepth=5,**kwards):
        super().__init__(in_channels, out_channels, kernel_size, padding,nheads, norm, norm_groups,ftdepth,**kwards)
        self.query  = Conv3DNormed( in_channels, out_channels,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)
        self.key    = Conv3DNormed( in_channels, out_channels,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)
        self.value  = Conv3DNormed( in_channels, out_channels,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)
        self.metric_channel = FTanimoto(depth=ftdepth, axis=[2,3,4])
        self.metric_space = FTanimoto(depth=ftdepth, axis=1)
        self.norm = get_norm(name=norm, axis=1, norm_groups= norm_groups)

class FTAttention2D(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=3, padding=1, nheads=1, norm = 'BatchNorm', norm_groups=None,ftdepth=5,**kwards):
        super().__init__(**kwards)

        self. att = RelFTAttention2D( in_channels, out_channels,kernel_size=kernel_size, padding=padding, nheads=nheads, norm = norm, norm_groups=norm_groups, ftdepth=ftdepth,**kwards)

    def forward(self, input):
        return self.att(input,input,input)

class CATFusion(nn.Module):
    """
    Alternative to concatenation followed by normed convolution: improves performance.
    """
    def __init__(self,nfilters_out,  in_channels_att, out_channels_att, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__(**kwards)
        # Or shall I use the same?
        self.relatt12 = RelFTAttention2D( in_channels_att, out_channels_att, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)
        self.relatt21 = RelFTAttention2D( in_channels_att, out_channels_att, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)
        self.fuse = Conv2DNormed(out_channels_att*2, nfilters_out,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads,**kwards)
        self.gamma1  = 0.1
        self.gamma2  = 0.1

    def forward(self, input_t1, input_t2):
        # These inputs must have the same dimensionality , t1, t2
        relatt12 = torch.mul(self.gamma1,self.relatt12(input_t1,input_t2,input_t2))
        relatt21 = torch.mul(self.gamma2,self.relatt21(input_t2,input_t1,input_t1))
        ones = torch.ones_like(input_t1)
        # Enhanced output of 1, based on memory of 2
        out12 = torch.mul(input_t1,ones + relatt12)
        # Enhanced output of 2, based on memory of 1
        out21 = torch.mul(input_t2,ones + relatt21)
        fuse = self.fuse(torch.cat((out12, out21),dim=1))
        fuse = F.relu(fuse)
        return fuse

class CATFusion3D(CATFusion):
    def __init__(self,nfilters_out,  in_channels_att, out_channels_att, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__(nfilters_out,  in_channels_att, out_channels_att, kernel_size, padding, nheads, norm, norm_groups, ftdepth,**kwards)
        # Or shall I use the same?
        self.relatt12 = RelFTAttention3D( in_channels_att, out_channels_att, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)
        self.relatt21 = RelFTAttention3D( in_channels_att, out_channels_att, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)
        self.fuse = Conv3DNormed(out_channels_att*2, nfilters_out,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads,**kwards)
        self.gamma1  = 0.1
        self.gamma2  = 0.1

class Fusion(CATFusion):
    def __init__(self,nfilters, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__(nfilters, nfilters, nfilters, kernel_size, padding,nheads, norm, norm_groups, ftdepth,**kwards)

####

class DownSample(nn.Module):
    def __init__(self, nfilters, factor=2,  _norm_type='BatchNorm', norm_groups=None, **kwargs):
        super().__init__(**kwargs)

        # Double the size of filters, since you downscale by 2.
        self.factor = factor
        self.nfilters = nfilters * self.factor

        self.kernel_size = (3,3)
        self.strides = (factor,factor)
        self.pad = (1,1)

        self.convdn = Conv2DNormed(self.nfilters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.pad,
                    _norm_type = _norm_type,
                    norm_groups=norm_groups)

    def forward(self,_xl):
        x = self.convdn(_xl)
        return x

class UpSample(nn.Module):
    def __init__(self,nfilters, factor = 2,  _norm_type='BatchNorm', norm_groups=None, **kwards):
        super().__init__(self,**kwards)


        self.factor = factor
        self.nfilters = nfilters // self.factor

        self.convup_normed = Conv2DNormed(self.nfilters,
                                              kernel_size = (1,1),
                                              _norm_type = _norm_type,
                                              norm_groups=norm_groups)

    def forward(self,_xl):
        x = F.UpSampling(_xl, scale=self.factor, sample_type='nearest')
        x = self.convup_normed(x)

        return x

class combine_layers(nn.Module):
    def __init__(self,_nfilters,  _norm_type = 'BatchNorm', norm_groups=None, **kwards):
        super().__init__(**kwards)
        # This performs convolution, no BatchNormalization. No need for bias.
        self.up = UpSample(_nfilters, _norm_type = _norm_type, norm_groups=norm_groups)
        self.conv_normed = Conv2DNormed(channels = _nfilters,
                                        kernel_size=(1,1),
                                        padding=(0,0),
                                        _norm_type=_norm_type,
                                        norm_groups=norm_groups)

    def forward(self,_layer_lo, _layer_hi):
        up = self.up(_layer_lo)
        up = F.relu(up)
        x = torch.cat((up,_layer_hi), dim=1)
        x = self.conv_normed(x)
        return x

class ResizeLayer(nn.Module):
    """
    Applies bilinear up/down sampling in spatial dims and changes number of filters as well
    """
    def __init__(self, nfilters, height, width,   _norm_type = 'BatchNorm', norm_groups=None, **kwards):

        self.height=height
        self.width = width
        self.up= nn.Upsample(size=(1,1,height, width), mode='bilinear')
        self.conv2d = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1, _norm_type=_norm_type, norm_groups = norm_groups, **kwards)


    def forward(self, input):
        #out = F.contrib.BilinearResize2D(input,height=self.height,width=self.width)
        out = self.up(input)
        out = self.conv2d(out)

        return out

class ExpandLayer(nn.Module):
    def __init__(self,nfilters, _norm_type = 'BatchNorm', norm_groups=None, ngroups=1,**kwards):
        super().__init__(**kwards)
        self.conv1 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups, _norm_type=_norm_type, norm_groups = norm_groups, **kwards)
        self.conv2 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)
        self.up= nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input):
        out = self.up(input)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)

        return out

class ExpandNCombine(nn.Module):
    def __init__(self,nfilters, _norm_type = 'BatchNorm', norm_groups=None,ngroups=1,**kwards):
        super().__init__(**kwards)
        self.conv1 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)
        self.conv2 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)
        self.up= nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input1, input2):
        out = self.up(input)
        out = self.conv1(out)
        out = F.relu(out)
        out2 = self.conv2(torch.cat((out,input2),dim=1))
        out2 = F.relu(out2)
        return out2

class CEEC_unit_v1(nn.Module):
    def __init__(self, nfilters, nheads= 1, ngroups=1, norm_type='BatchNorm', norm_groups=None, ftdepth=5, **kwards):
        super().__init__(**kwards)
        nfilters_init = nfilters//2
        self.conv_init_1 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)
        self.compr11 = Conv2DNormed(channels=nfilters_init*2, kernel_size=3,padding=1,strides=2, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)
        self.compr12 = Conv2DNormed(channels=nfilters_init*2, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)
        self.expand1 = ExpandNCombine(nfilters_init,_norm_type = norm_type, norm_groups=norm_groups,ngroups=ngroups)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.conv_init_2 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)#half size
        self.expand2 = ExpandLayer(nfilters_init//2 ,_norm_type = norm_type, norm_groups=norm_groups,ngroups=ngroups )
        self.compr21 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=2, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)
        self.compr22 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)
        # Will join with master input with concatenation  -- IMPORTANT: ngroups = 1 !!!!
        self.collect = Conv2DNormed(channels=nfilters, kernel_size=3,padding=1,strides=1, groups=1, _norm_type=norm_type, norm_groups=norm_groups,**kwards)
        self.att  = FTAttention2D(nkeys=nfilters,nheads=nheads,norm=norm_type, norm_groups = norm_groups,ftdepth=ftdepth)
        self.ratt122 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads,norm=norm_type, norm_groups = norm_groups,ftdepth=ftdepth)
        self.ratt211 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads,norm=norm_type, norm_groups = norm_groups,ftdepth=ftdepth)
        self.gamma1  = 0.1
        self.gamma2  = 0.1
        self.gamma3  = 0.1


    def forward(self, input):

        # =========== UNet branch ===========
        out10 = self.conv_init_1(input)
        out1 = self.compr11(out10)
        out1 = F.relu(out1)
        out1 = self.compr12(out1)
        out1 = F.relu(out1)
        out1 = self.expand1(out1,out10)
        out1 = F.relu(out1)
        # =========== \capNet branch ===========
        #input = F.identity(input) # Solves a mxnet bug
        out20 = self.conv_init_2(input)
        out2 = self.expand2(out20)
        out2 = F.relu(out2)
        out2 = self.compr21(out2)
        out2 = F.relu(out2)
        out2 = self.compr22(torch.cat((out2,out20),dim=1))
        out2 = F.relu(out2)
        att  = torch.mul(self.gamma1,self.att(input))
        ratt122 = torch.mul(self.gamma2,self.ratt122(out1,out2,out2))
        ratt211 = torch.mul(self.gamma3,self.ratt211(out2,out1,out1))
        ones1 = torch.ones_like(out10)
        ones2 = torch.ones_like(input)
        # Enhanced output of 1, based on memory of 2
        out122 = torch.mul(out1,ones1 + ratt122)
        # Enhanced output of 2, based on memory of 1
        out211 = torch.mul(out2,ones1 + ratt211)
        out12 = F.relu(self.collect(torch.cat((out122,out211),dim=1)))
        # Emphasize residual output from memory on input
        out_res = torch.mul(input + out12, ones2 + att)
        return out_res

class combine_layers_wthFusion(nn.Module):
    def __init__(self,nfilters, nheads=1,  _norm_type = 'BatchNorm', norm_groups=None,ftdepth=5, **kwards):
        super().__init__(self,**kwards)
        self.conv1 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1, groups=nheads, _norm_type=_norm_type, norm_groups = norm_groups, **kwards)# restore help
        self.conv3 = Fusion(nfilters=nfilters, kernel_size=3, padding=1, nheads=nheads, norm=_norm_type, norm_groups = norm_groups, ftdepth=ftdepth,**kwards) # process
        self.up= nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self,_layer_lo, _layer_hi):
        up = self.up(_layer_lo)
        up = self.conv1(up)
        up = F.relu(up)
        x = self.conv3(up,_layer_hi)
        return x

class ExpandNCombine_V3(nn.Module):
    def __init__(self,nfilters, _norm_type = 'BatchNorm', norm_groups=None,ngroups=1,ftdepth=5,**kwards):
        super().__init__(**kwards)
        self.conv1 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)# restore help
        self.conv2 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)# restore help
        self.conv3 = Fusion(nfilters=nfilters,kernel_size=3,padding=1,nheads=ngroups,norm=_norm_type, norm_groups = norm_groups,ftdepth=ftdepth,**kwards) # process
        self.up= nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, F, input1, input2):
        out =self.up(input1)
        out = self.conv1(out)
        out1 = F.relu(out)
        out2 = self.conv2(input2)
        out2 = F.relu(out2)
        outf = self.conv3(out1,out2)
        outf = F.relu(outf)
        return outf

class CEEC_unit_v2(nn.Module):
    def __init__(self, nfilters, nheads= 1, ngroups=1, norm_type='BatchNorm', norm_groups=None, ftdepth=5, **kwards):
        super().__init__(**kwards)

        nfilters_init = nfilters//2
        self.conv_init_1 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)#half size
        self.compr11 = Conv2DNormed(channels=nfilters_init*2, kernel_size=3,padding=1,strides=2, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)#half size
        self.compr12 = Conv2DNormed(channels=nfilters_init*2, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)# process
        self.expand1 = ExpandNCombine_V3(nfilters_init,_norm_type = norm_type, norm_groups=norm_groups,ngroups=ngroups,ftdepth=ftdepth) # restore original size + process
        self.conv_init_2 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)#half size
        self.expand2 = ExpandLayer(nfilters_init//2 ,_norm_type = norm_type, norm_groups=norm_groups,ngroups=ngroups )
        self.compr21 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=2, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)
        self.compr22 = Fusion(nfilters=nfilters_init, kernel_size=3,padding=1, nheads=ngroups, norm=norm_type, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)
        self.collect = CATFusion(nfilters_out=nfilters, nfilters_in=nfilters_init, kernel_size=3,padding=1,nheads=1, norm=norm_type, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)
        self.att  = FTAttention2D(nkeys=nfilters,nheads=nheads,norm=norm_type, norm_groups = norm_groups, ftdepth=ftdepth)
        self.ratt122 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads,norm=norm_type, norm_groups = norm_groups, ftdepth=ftdepth)
        self.ratt211 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads,norm=norm_type, norm_groups = norm_groups, ftdepth=ftdepth)
        self.gamma1  = 0.1
        self.gamma2  = 0.1
        self.gamma3  = 0.1

    def forward(self, input):
        # =========== UNet branch ===========
        out10 = self.conv_init_1(input)
        out1 = self.compr11(out10)
        out1 = F.relu(out1)
        #print (out1.shape)
        out1 = self.compr12(out1)
        out1 = F.relu(out1)
        #print (out1.shape)
        out1 = self.expand1(out1,out10)
        out1 = F.relu(out1)
        # =========== \capNet branch ===========
        #input = F.identity(input) # Solves a mxnet bug
        out20 = self.conv_init_2(input)
        out2 = self.expand2(out20)
        out2 = F.relu(out2)
        out2 = self.compr21(out2)
        out2 = F.relu(out2)
        out2 = self.compr22(out2,out20)
        #input = F.identity(input) # Solves a mxnet bug
        att  = torch.mul(self.gamma1,self.att(input))
        ratt122 = torch.mul(self.gamma2,self.ratt122(out1,out2,out2))
        ratt211 = torch.mul(self.gamma3,self.ratt211(out2,out1,out1))
        ones1 = torch.ones_like(out10)
        ones2 = torch.ones_like(input)
        # Enhanced output of 1, based on memory of 2
        out122 = torch.mul(out1,ones1 + ratt122)
        # Enhanced output of 2, based on memory of 1
        out211 = torch.mul(out2,ones1 + ratt211)
        out12 = self.collect(out122,out211) # includes relu, it's for fusion
        out_res = torch.mul(input + out12, ones2 + att)
        return out_res

class ResNet_v2_block(nn.Module):
    """
    ResNet v2 building block. It is built upon the assumption of ODD kernel
    """
    def __init__(self, _nfilters,_kernel_size=(3,3),_dilation_rate=(1,1),
                 _norm_type='BatchNorm', norm_groups=None, ngroups=1, **kwards):
        super().__init__(**kwards)
        self.nfilters = _nfilters
        self.kernel_size = _kernel_size
        self.dilation_rate = _dilation_rate
        # Ensures padding = 'SAME' for ODD kernel selection
        p0 = self.dilation_rate[0] * (self.kernel_size[0] - 1)/2
        p1 = self.dilation_rate[1] * (self.kernel_size[1] - 1)/2
        p = (int(p0),int(p1))
        self.BN1 = get_norm(_norm_type, norm_groups=norm_groups )
        self.conv1 = torch.nn.Conv2d(self.nfilters,kernel_size = self.kernel_size,padding=p,dilation=self.dilation_rate,bias=False,groups=ngroups)
        self.BN2 = get_norm(_norm_type, norm_groups= norm_groups)
        self.conv2 = torch.nn.Conv2d(self.nfilters,kernel_size = self.kernel_size,padding=p,dilation=self.dilation_rate,bias=True, groups=ngroups)

    def forward(self,_input_layer):
        x = self.BN1(_input_layer)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x

class FracTALResNet_unit(nn.Module):
    def __init__(self, nfilters, ngroups=1, nheads=1, kernel_size=(3,3), dilation_rate=(1,1), norm_type = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__(**kwards)
        self.block1 = ResNet_v2_block(nfilters,kernel_size,dilation_rate,_norm_type = norm_type, norm_groups=norm_groups, ngroups=ngroups)
        self.attn = FTAttention2D(nkeys=nfilters, nheads=nheads, kernel_size=kernel_size, norm = norm_type, norm_groups = norm_groups,ftdepth=ftdepth)
        self.gamma  = 0.1

    def forward(self, input):
        out1 = self.block1(input)
        att = self.attn(input)
        att= torch.mul(self.gamma,att)
        out  = torch.mul((input + out1) , torch.ones_like(out1) + att)
        return out
