import math
from model.transformer import PositionalEncoding
import torch
import torch.nn.functional as F
from torch import nn
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from torchvision import transforms
from .transformer_v2 import PositionalEncoding
from .transformer_v2 import InfoTransformer
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np
from .stylegan import EqualLinear,PixelNorm,StyledContentBlock
SHUT_BN = False


class StyleSRN(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 n_mlp=8,
                 n_eca=3,
                 mask=True,
                 hidden_units=32,
                 text_emb=95,
                 out_text_channels=64,
                 feature_rotate=False,
                 rotate_train=3.):
        super(StyleSRN, self).__init__()

        self.text_emb=text_emb
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )

        self.style_encoder=StyleEncoderChannel(n_mlp,n_eca)   # Style Embedding Branch
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            # Style Content Fuse Block
            setattr(self, 'block%d' % (i + 2), StyleContentFuse(hidden_units, out_text_channels))
        # InfoGen(text_emb, out_text_channels)
        self.infoGen = TPInterpreter(text_emb, out_text_channels, output_size=(height//scale_factor, width//scale_factor))
        self.feature_rotate = feature_rotate
        self.rotate_train = rotate_train

        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)

        self.block_range = [k for k in range(2, self.srb_nums + 2)]

    def forward(self, x, text_emb=None):
        style_vec=self.style_encoder(x[:,:3,:,:]) # get style vector w
        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        if text_emb is None:
            text_emb = torch.zeros(1, self.text_emb, 1, 26).to(x.device)
        padding_feature = block['1']

        tp_map= self.infoGen(padding_feature, text_emb)

        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                # Style Content Fuse Block forward
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)],style_vec,tp_map)

            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)])) #
        output = torch.tanh(block[str(self.srb_nums + 3)])

        return output


class StyleEncoder(nn.Module):
    def __init__(self, n_mlp,out_channels=64):
        super(StyleEncoder, self).__init__()
        resnet=models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Remove the final FC layer
        self.fc = nn.Linear(resnet.fc.in_features, out_channels)
        self.map=StyleMappingNet(n_mlp)

    def forward(self,x):
        y = self.features(x)
        y = y.view(y.size(0), -1)
        z = self.fc(y)
        w =self.map(z)
        return w


class StyleEncoderChannel(nn.Module):
    def __init__(self, n_mlp,n_eca,out_channels=64,kernel_size=3):
        super(StyleEncoderChannel, self).__init__()
        resnet=models.resnet18(pretrained=True)
        self.block1 = nn.Sequential(*list(resnet.children())[:-2])
        for i in range(n_eca):
            setattr(self, 'block%d' % (i + 2), ECA(resnet.fc.in_features,kernel_size)) 
        self.fc = nn.Linear(resnet.fc.in_features, out_channels)
        self.map=StyleMappingNet(n_mlp)
        self.n_eca=n_eca

    def forward(self,x):
        block = {'1': self.block1(x)}
        for i in range(self.n_eca):
          block[str(i+2)] = getattr(self, 'block%d' % (i + 2))(block[str(i+1)])
        feature=block[str(i+2)]
        style_vec=F.adaptive_avg_pool2d(feature, (1,1))
        style_vec = style_vec.view(style_vec.size(0), -1)
        z = self.fc(style_vec)
        w =self.map(z)
        return w


class StyleContentFuse(nn.Module):
    def __init__(self, hidden_units,out_text_channels):
        super().__init__()
        self.style = StyledContentBlock(2 * hidden_units, 2 * hidden_units)
        self.content = RecurrentResidualBlockTL(2 * hidden_units, out_text_channels)

    def forward(self,x,style,text_emb):
        x=self.style(x,style)
        x=self.content(x,text_emb)
        return x


class StyleMappingNet(nn.Module):
    def __init__(self, n_mlp,code_dim=64):
        super().__init__()
        
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        # mapping network f, used to generate the latent vector W from the noise vector Z
        self.style = nn.Sequential(*layers)

    def forward(self, x):
        x = self.style(x)
        return x


class ECA(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ECA, self).__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              groups=in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Average pooling yields global statistics for each channel.
        y = torch.mean(x, dim=[2, 3], keepdim=True)
        y = y.view(batch_size, channels, -1)  # [batch_size, channels, 1]

        # Compute channel weights using 1D convolution
        y = self.conv(y)

        y = self.sigmoid(y)
        y = y.unsqueeze(-1)
        y = y.expand(-1, -1, height, width)  # [batch_size, channels, height, width]
        # Apply Attention Weights
        return x * y


class RecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTL, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        # self.conv_proj = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)
        self.gru1 = GruBlock(channels + text_channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.gru2 = GruBlock(channels, channels)
        self.gru2 = GruBlock(channels, channels) # + text_channels

        # self.concat_conv = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)

    def forward(self, x, text_emb):

        residual = self.conv1(x)
        if not SHUT_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if not SHUT_BN:
            residual = self.bn2(residual)

        ############ Fusing with TL ############
        cat_feature = torch.cat([residual, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################
        # fused_feature = self.conv_proj(cat_feature)
        residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)


class TPInterpreter(nn.Module):
    def __init__(
                self,
                t_emb,
                out_text_channels,
                output_size=(16, 64),
                feature_in=64,
                # d_model=512,
                t_encoder_num=1,
                t_decoder_num=2,
                 ):
        super(TPInterpreter, self).__init__()

        d_model = out_text_channels # * output_size[0]
        self.fc_in = nn.Linear(t_emb, d_model)
        self.fc_feature_in = nn.Linear(feature_in, d_model)

        self.activation = nn.PReLU()

        self.transformer = InfoTransformer(d_model=d_model,
                                          dropout=0.1,
                                          nhead=4,
                                          dim_feedforward=d_model,
                                          num_encoder_layers=t_encoder_num,
                                          num_decoder_layers=t_decoder_num,
                                          normalize_before=False,
                                          return_intermediate_dec=True, feat_height=output_size[0], feat_width=output_size[1])

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)

        self.output_size = output_size
        self.seq_len = output_size[1] * output_size[0] #output_size[1] ** 2 #
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.masking = torch.ones(output_size)

        # self.tp_uper = InfoGen(t_emb, out_text_channels)

    def forward(self, image_feature, tp_input):

        # H, W = self.output_size
        x = tp_input

        N_i, C_i, H_i, W_i = image_feature.shape
        H, W = H_i, W_i


        x_tar = image_feature

        # [1024, N, 64]
        x_im = x_tar.view(N_i, C_i, H_i * W_i).permute(2, 0, 1)

        device = x.device
        x = x.permute(0, 3, 1, 2).squeeze(-1)
        x = self.activation(self.fc_in(x))
        N, L, C = x.shape

        x_pos = self.pe(torch.zeros((N, L, C)).to(device)).permute(1, 0, 2)
        mask = torch.zeros((N, L)).to(device).bool()
        x = x.permute(1, 0, 2)

        text_prior, pr_weights = self.transformer(x, mask, self.init_factor.weight, x_pos, tgt=x_im, spatial_size=(H, W)) # self.init_factor.weight
        text_prior = text_prior.mean(0)

        # [N, L, C] -> [N, C, H, W]
        text_prior = text_prior.permute(1, 2, 0).view(N, C, H, W)

        return text_prior


class ContentEmbedding(nn.Module):
    def __init__(
                self,
                t_emb,
                out_text_channels,
                output_size=(16, 64),
                feature_in=64,
                t_encoder_num=1,
                t_decoder_num=2,
                 ):
        super(ContentEmbedding, self).__init__()

        d_model = out_text_channels
        self.fc_in = nn.Linear(t_emb, d_model)
        self.fc_feature_in = nn.Linear(feature_in, d_model)

        self.activation = nn.PReLU()

        # 图像特征的Transformer编码器
        self.img_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model),
            num_layers=t_encoder_num
        )

        # 文本先验的Transformer编码器
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model),
            num_layers=t_encoder_num
        )

        # 融合的双向Transformer解码器
        self.bidir_transformer = nn.Transformer(
            d_model=d_model,
            nhead=4,
            num_encoder_layers=t_encoder_num,
            num_decoder_layers=t_decoder_num,
            dim_feedforward=d_model,
            dropout=0.1
        )

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)
        self.output_size = output_size
        self.seq_len = output_size[1] * output_size[0]
        self.init_factor = nn.Embedding(self.seq_len, d_model)

    def forward(self, image_feature, tp_input):
        x = tp_input

        N_i, C_i, H_i, W_i = image_feature.shape
        H, W = H_i, W_i

        # 图像特征编码
        x_im = image_feature.view(N_i, C_i, H_i * W_i).permute(2, 0, 1)
        img_encoded = self.img_transformer(x_im)

        device = x.device
        x = x.permute(0, 3, 1, 2).squeeze(-1)
        x = self.activation(self.fc_in(x))
        x_pos = self.pe(torch.zeros((x.size(0), x.size(1), x.size(2))).to(device)).permute(1, 0, 2)
        mask = torch.zeros((x.size(0), x.size(1))).to(device).bool()
        x = x.permute(1, 0, 2)

        # 文本特征编码
        text_encoded = self.text_transformer(x)

        # 双向Transformer解码器，交互文本和图像特征
        text_prior = self.bidir_transformer(text_encoded, img_encoded)

        text_prior = text_prior.permute(1, 2, 0).view(N_i, C_i, H, W)

        return text_prior, None


def unpatchify(x, patch_size: int = 4, shape: tuple = (32, 128)):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = shape[0] // p
    w = shape[1] // p
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs

def showPIL(img,batch_id=0):
    img = img[batch_id,:3,...]*255
    img = torch.as_tensor(img.detach().cpu(),dtype=torch.uint8).numpy()
    img = transforms.functional.to_pil_image(img.transpose((1,2,0)))
    img.show()


def show_batch_images(img_batch):
    img_batch = img_batch[:, :3, ...] * 255

    img_batch = torch.as_tensor(img_batch.detach().cpu(), dtype=torch.uint8).numpy()
    grid_size = int(np.ceil(np.sqrt(img_batch.shape[0])))
    fig, axarr = plt.subplots(grid_size,grid_size, figsize=(10, 10))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < img_batch.shape[0]:
                img = img_batch[idx].transpose(1, 2, 0)
                axarr[i, j].imshow(img)
                axarr[i, j].axis('off')
            else:
                axarr[i, j].axis('off')

class AffineModulate(nn.Module):
    def __init__(self, channel=64):
        super(AffineModulate, self).__init__()
        self.BN = nn.BatchNorm2d(channel)
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 1)
        )
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 1)
        )
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel, channel, 1)
        )
        self.channel_attn = nn.Sequential(
            # Linear
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(16, 64), groups=64),  # 全局深度可分离
            nn.Conv2d(channel, channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1),
            nn.Sigmoid()
        )
        self.aggregation = nn.Sequential(
            nn.Conv2d(channel,channel,1),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            nn.Conv2d(channel,channel,1)
        )
        self.BN2 = nn.BatchNorm2d(channel)

    def forward(self,image_feature,tp_map):
        tp_map = self.BN(tp_map)
        cat_feature = torch.cat([image_feature,tp_map],dim=1)
        x1,x2,x3 = self.conv1x1_1(cat_feature),self.conv1x1_2(cat_feature),self.conv1x1_3(cat_feature)
        x1 = self.channel_attn(x1)
        x2 = self.aggregation(x1*x2)
        return self.BN2(x3+x2)



class RecurrentResidualBlockAffine(nn.Module):
    def __init__(self, channels,text_channel=None):
        # channls = 64
        super(RecurrentResidualBlockAffine, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels) # + text_channels
        self.affine = AffineModulate(channel=channels)
        self.BN = nn.BatchNorm2d(channels)


    def forward(self, x, tp_map):
        residual = self.conv1(x)
        if not SHUT_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if not SHUT_BN:
            residual = self.bn2(residual)
        # AdaFM
        residual = self.affine(residual,tp_map)
        # SRB
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)

        return self.gru2(self.BN(x + residual))




class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class TSRN_TL_TRANS(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 text_emb=37, #37, #26+26+1 3965
                 out_text_channels=64, # 32 256
                 feature_rotate=False,
                 rotate_train=3.):
        super(TSRN_TL_TRANS, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockTL(2 * hidden_units, out_text_channels)) #RecurrentResidualBlockTL

        self.infoGen = TPInterpreter(text_emb, out_text_channels, output_size=(height//scale_factor, width//scale_factor)) # InfoGen(text_emb, out_text_channels)

        self.feature_rotate = feature_rotate
        self.rotate_train = rotate_train

        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)

        self.block_range = [k for k in range(2, self.srb_nums + 2)]

    # print("self.block_range:", self.block_range)

    def forward(self, x, text_emb=None, text_emb_gt=None, feature_arcs=None, rand_offs=None):

        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        if text_emb is None:
            text_emb = torch.zeros(1, 37, 1, 26).to(x.device) #37
        padding_feature = block['1']

        tp_map_gt, pr_weights_gt = None, None
        tp_map, pr_weights = self.infoGen(padding_feature, text_emb)

        # N, C, H, W

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                # pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                # all_pred_vecs.append(pred_word_vecs)
                # if not self.training:
                #     word_vecs = pred_word_vecs

                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], tp_map)

            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])


        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)])) #

        output = torch.tanh(block[str(self.srb_nums + 3)])

        self.block = block

        if self.training:
            ret_mid = {
                "pr_weights": pr_weights,
                "pr_weights_gt": pr_weights_gt,
                "spatial_t_emb": tp_map,
                "spatial_t_emb_gt": tp_map_gt,
                "in_feat": block["1"],
                "trans_feat": tp_map
            }

            return output, ret_mid

        else:
            return output, pr_weights

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, 16, 64).to(device)
    model=StyleEncoder()
    out=model(x)
    print(out.shape)