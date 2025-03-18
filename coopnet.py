"""
@File: COOPNet.py
@Time: 2024/12/20
@Author: TL
@Software: PyCharm

"""

""""
backbone is SwinTransformer
"""
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from models.SwinT import SwinTransformer
from models.vgg import LowFeatureExtract

import torch.nn.functional as F
import os
import onnx

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class COOPNet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(COOPNet, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        self.hff4 = HFF(1024, 12, 12, 4)  # 多模态融合
        self.hff3 = HFF(512, 24, 24, 4)
        self.lff2 = LFF(256, 48, 48, 4)
        self.lff1 = LFF(128, 96, 96, 4)

        self.decoder = Decoder()
        self.decoder2 = Decoder()
        self.con3x3 = conv3x3_bn_relu(256,256)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up16 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.conv256_32 = conv3x3_bn_relu(256, 32)
        self.conv512_32 = conv3x3_bn_relu(512, 32)
        self.conv1024_32 = conv3x3_bn_relu(1024, 32)
        self.conv64_1 = conv3x3(64, 1)

        self.edge_layer = Edge_Module()
        self.edge_feature = conv3x3_bn_relu(1, 32)
        self.fuse_edge_sal = conv3x3(32, 1)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            conv3x3(32, 1)
        )

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dwc3 = conv3x3_bn_relu(512, 1024)
        self.dwc2 = conv3x3_bn_relu(256, 512)
        self.dwc1 = conv3x3_bn_relu(128, 256)
        self.dwcon_1 = conv3x3_bn_relu(512, 256)
        self.dwcon_2 = conv3x3_bn_relu(1024, 512)
        self.dwcon_3 = conv3x3_bn_relu(2048, 1024)
        self.dwcon_4 = conv3x3_bn_relu(256, 128)
        self.conv128_256 = conv3x3_bn_relu(128, 256)
        self.conv64_256 = conv3x3_bn_relu(64, 256)
        self.conv1024_256 = conv3x3_bn_relu(2048, 256)
        self.relu = nn.ReLU(True)
        self.space_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(256)
        self.low_feature_extract = LowFeatureExtract()
        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
    def forward(self, x, d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r1 = rgb_list[0]  # 128*96
        r2 = rgb_list[1]  # 256*48
        r3 = rgb_list[2]  # 512*24
        r4 = rgb_list[3]  # 1024*12

        d1 = depth_list[0]
        d2 = depth_list[1]
        d3 = depth_list[2]
        d4 = depth_list[3]

        r1_down = self.dwc1(self.down(r1))  # 256  48
        r2_down = self.dwc2(self.down(r2))  # 512  24
        r3_down = self.dwc3(self.down(r3))  # 1024 12

        d1_down = self.dwc1(self.down(d1))
        d2_down = self.dwc2(self.down(d2))
        d3_down = self.dwc3(self.down(d3))


        feature_224, feature_112 = self.low_feature_extract(x)
        f112 = F.interpolate(feature_112, size=(96, 96), mode='bilinear')
        # print(feature_112.shape)       torch.Size([7, 128, 192, 192])
                                        # torch.Size([7, 64, 384, 384])
        # print(feature_224.shape)
        f112 = self.conv128_256(f112)
        # sa112 = self.space_attention(f112)
        f224 = F.interpolate(feature_224, size=(96, 96), mode='bilinear')
        f224 = self.conv64_256(f224)
        # sa224 = self.space_attention(f224)

        r1_down_r2 = self.dwcon_1(torch.cat((r1_down, r2), 1))  # 512-->    256 48
        r2_down_r3 = self.dwcon_2(torch.cat((r2_down, r3), 1))  # 1024-->   512 24
        r3_down_r4 = self.dwcon_3(torch.cat((r3_down, r4), 1))  # 2048-->   1024 12

        d1_down_d2 = self.dwcon_1(torch.cat((d1_down, d2), 1))  # 512-->256 48
        d2_down_d3 = self.dwcon_2(torch.cat((d2_down, d3), 1))  # 1024-->512 24
        d3_down_d4 = self.dwcon_3(torch.cat((d3_down, d4), 1))  # 2048-->1024 12

        fuse1 = self.lff1(r1, d1)  # [256]  96
        fuse1_down = self.down(fuse1)  # [256]  48

        fuse2 = self.lff2(r1_down_r2, d1_down_d2, fuse1_down)  # [512]  48
        fuse2_down = self.down(fuse2)  # [512] 24

        fuse3 = self.hff3(r2_down_r3, d2_down_d3, fuse2_down)  # [1024] 24
        fuse3_down = self.down(fuse3)  # [1024] 12

        fuse4 = self.hff4(r3_down_r4, d3_down_d4, fuse3_down)  # [2048] 12

        end_fuse1, out43, out432 = self.decoder(fuse4, fuse3, fuse2, fuse1)


        end_fuse, end_fuse2, end_fuse3 = self.decoder2(fuse4, out43, out432, end_fuse1, end_fuse1)
        end = self.fuse(torch.cat((end_fuse,f112),1))
        end1 = self.channel_attention(end) * end
        end1 = end1 + end
         

        r4_up = F.interpolate(fuse4, size=(96, 96), mode='bilinear')
        r4_up = self.conv1024_256(r4_up)
        end3 = self.fuse(torch.cat((r4_up, end_fuse), 1))
        end2 = self.channel_attention(end3) * end3
        end2 = end2 + end3


        end_fuse = self.fuse1(torch.cat((end1, end2, end_fuse,end_fuse1), 1))
        

        edge_map = self.edge_layer(d1, d2)
        edge_feature = self.edge_feature(edge_map)
        edge_feature2 = F.interpolate(edge_feature, size=(24, 24), mode='bilinear')
        edge_feature3 = F.interpolate(edge_feature, size=(48, 48), mode='bilinear')
        end_sal = self.conv256_32(end_fuse)  # [b,32]
        end_sal1 = self.conv256_32(end_fuse1)
        end_sal2 = self.conv1024_32(end_fuse2)
        end_sal3 = self.conv512_32(end_fuse3)
        up_edge = self.up_edge(edge_feature)
        out3 = self.relu(torch.cat((end_sal3, edge_feature3), dim=1))
        out2 = self.relu(torch.cat((end_sal2, edge_feature2), dim=1))
        out1 = self.relu(torch.cat((end_sal1, edge_feature), dim=1))
        out = self.relu(torch.cat((end_sal, edge_feature), dim=1))
        out = self.up4(out)
        out1 = self.up4(out1)
        out2 = self.up16(out2)
        out3 = self.up8(out3)
        sal_out = self.conv64_1(out)
        sal_out1 = self.conv64_1(out1)
        sal_out2 = self.conv64_1(out2)
        sal_out3 = self.conv64_1(out3)

        return sal_out, up_edge, sal_out1, sal_out2, sal_out3

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")

    # def load_pre(self, model):
    #     self.edge_layer.load_state_dict(model,strict=False)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False, ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias, )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
            self,
            gate_channels,
            reduction_ratio=16,
            pool_types=["avg", "max"],
            no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out

        return out


class LFF(nn.Module):
    def __init__(self, infeature, w=12, h=12, heads=4):
        super(LFF, self).__init__()
        self.conv = conv3x3_bn_relu(infeature, infeature)  # 3x3 去噪
        dim = infeature * 2
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.PReLU()
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(infeature * 3, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.PReLU()
        )
        self.channel_attention = ChannelAttention(infeature)
        # self.spatial_attention = SpatialAttention()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.tri_att = TripletAttention(infeature)

    def forward(self, r, d, rd=None):
        # r_sa =r * self.tri_att(r)
        # d_sa =d * self.tri_att(d)
        r_triatt = self.tri_att(r)
        d_triatt = self.tri_att(d)

        r1 = self.mean(r)
        r2 = self.max_pool(r)

        d1 = self.mean(d)
        d2 = self.max_pool(d)

        r_r1 = r * self.sigmoid(r1)
        r_r2 = r * self.sigmoid(r2)

        d_d1 = d * self.sigmoid(d1)
        d_d2 = d * self.sigmoid(d2)

        r1_r2 = self.conv(r_r1 * r_r2 + r)
        d1_d2 = self.conv(d_d1 * d_d2 + d)

        r_d = r1_r2 * d
        d_r = d1_d2 * r

        ca_r_d = torch.matmul(r_d, r_triatt)
        ca_d_r = torch.matmul(d_r, d_triatt)

        out = self.fuse(torch.cat((ca_r_d, ca_d_r), 1))  # C: infeature *2
        if rd is None:
            return out  # 256*96 lff1
        else:

            out = self.fuse1(torch.cat((rd, out), 1))  # 512*48    512*48  ---256*48
            return out  # 512 * 48 lff2


class HFF(nn.Module):
    def __init__(self, infeature, w=12, h=12, heads=4):
        super(HFF, self).__init__()

        down_dim = infeature // 2
        self.conv = conv3x3_bn_relu(down_dim, down_dim)
        self.down_conv = nn.Sequential(
            nn.Conv2d(infeature, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )  # 1x1 通道减半

        self.conv1 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.PReLU()
        )

        self.query_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim // 8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(2 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(2 * down_dim, 2 * infeature, kernel_size=1), nn.BatchNorm2d(2 * infeature), nn.PReLU()
        )
        self.softmax = nn.Softmax(dim=-1)
        self.channel_attention = ChannelAttention(down_dim)
        self.spatial_attention = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

    def forward(self, r, d, rd):
        r = self.down_conv(r)
        d = self.down_conv(d)  # infeature // 2
        rd = self.down_conv(rd)

        out_r1 = self.conv1(r)
        out_r2 = self.conv2(r)
        out_r3 = self.conv3(r)
        out_r4 = self.conv4(r)
        out_r5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(r, 1)), size=r.size()[2:],
                            mode='bilinear')
        ASPP_r = self.fuse(torch.cat((out_r1, out_r2, out_r3, out_r4, out_r5), 1))

        out_d1 = self.conv1(d)
        out_d2 = self.conv2(d)
        out_d3 = self.conv3(d)
        out_d4 = self.conv4(d)
        out_d5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(d, 1)), size=d.size()[2:],
                            mode='bilinear')
        ASPP_d = self.fuse(torch.cat((out_d1, out_d2, out_d3, out_d4, out_d5), 1))

        rd_ca = self.sigmoid(self.conv(rd * self.channel_attention(rd)))
        rd_sa = self.conv(rd * self.spatial_attention(rd))

        r1 = r * rd_sa
        d1 = d * rd_sa

        # C: infeature // 2

        conv_r = self.fuse1(torch.cat((ASPP_r, r1), 1))
        conv_d = self.fuse1(torch.cat((ASPP_d, d1), 1))

        m_batchsize, C, height, width = r.size()  # C: infeature // 2
        proj_query = self.query_conv2(conv_r).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv2(conv_d).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv2(conv_d).view(m_batchsize, -1, width * height)
        out1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)  # C: infeature // 2

        out_r = self.gamma2 * out1 + r

        m_batchsize, C, height, width = d.size()  # C: infeature // 2
        proj_query1 = self.query_conv2(conv_d).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key1 = self.key_conv2(conv_r).view(m_batchsize, -1, width * height)
        energy1 = torch.bmm(proj_query1, proj_key1)
        attention1 = self.softmax(energy1)
        proj_value1 = self.value_conv2(conv_r).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value1, attention1.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)  # C: infeature // 2

        out_d = self.gamma2 * out2 + d  # C: infeature // 2

        out_r = out_r * rd_ca
        out_d = out_d * rd_ca
        # ----------------------
        sa = self.spatial_attention(out_r * out_d)
        out_d_f = sa * out_d
        out_d_f = out_d_f + out_d
        out_d_ca = self.channel_attention(out_d_f)
        d_out = out_d * out_d_ca

        out_r_f = sa * out_r
        out_r_f = out_r_f + out_r
        out_r_ca = self.channel_attention(out_r_f)
        r_out = out_r * out_r_ca
        # ----------------------
        r2 = r_out + d_out
        d2 = r_out * d_out

        out = self.fuse2(torch.cat((r2, d2), 1))  # C: infeature
        return out


class MSFA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSFA, self).__init__()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = conv3x3_bn_relu(in_ch, out_ch)
        self.aff = AFF(out_ch)

    def forward(self, fuse_high, fuse_low):
        fuse_high = self.up2(fuse_high)
        fuse_high = self.conv(fuse_high)
        fe_decode = self.aff(fuse_high, fuse_low)
        # fe_decode = fuse_high + fuse_low
        return fe_decode


# cfm = CFM(2048, 1024)
# a = torch.randn([1, 2048, 12, 12])
# b = torch.randn([1, 1024, 24, 24])
# re1 = cfm(a, b)
# print("re1.shape:", re1.shape)
#
# cfm1 = CFM(1024, 512)
# c = torch.randn([1, 512, 48, 48])
# re2 = cfm1(re1, c)
# print("re2.shape:", re2.shape)
# cfm2 = CFM(512, 256)
# d = torch.randn([1, 256, 96, 96])
# re3 = cfm2(re2, d)
# print("re3.shape:", re3.shape)

# Cascaded Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm12 = MSFA(2048, 1024)
        self.cfm23 = MSFA(1024, 512)
        self.cfm34 = MSFA(512, 256)
        self.conv256_512 = conv3x3_bn_relu(256, 512)
        self.conv256_1024 = conv3x3_bn_relu(256, 1024)
        self.conv256_2048 = conv3x3_bn_relu(256, 2048)

        """
        此处参数：fuse1,fuse2,fuse3,fuse4 特征等级依次升高，通道数逐渐升高，尺寸逐渐减小
        fuse4:2048,12,12
        fuse3:1024,24,24
        fuse2:512,48,48
        fuse1:256,96,96
        out为上一个解码器预测的:1,256,96,96
        """

    def forward(self, fuse4, fuse3, fuse2, fuse1, iter=None):
        if iter is not None:

            out_fuse4 = F.interpolate(iter, size=(12, 12), mode='bilinear')
            out_fuse4 = self.conv256_2048(out_fuse4)
            fuse4 = out_fuse4 + fuse4

            out_fuse3 = F.interpolate(iter, size=(24, 24), mode='bilinear')
            out_fuse3 = self.conv256_1024(out_fuse3)
            fuse3 = out_fuse3 + fuse3

            out_fuse2 = F.interpolate(iter, size=(48, 48), mode='bilinear')
            out_fuse2 = self.conv256_512(out_fuse2)
            fuse2 = out_fuse2 + fuse2

            fuse1 = iter + fuse1

            out43 = self.cfm12(fuse4, fuse3)
            out432 = self.cfm23(out43, fuse2)
            out4321 = self.cfm34(out432, fuse1)
            return out4321, out43, out432
        else:
            out43 = self.cfm12(fuse4, fuse3)  # [b,1024,24,24]
            out432 = self.cfm23(out43, fuse2)  # [b,512,48,48]
            out4321 = self.cfm34(out432, fuse1)  # [b,256,96,96]
            return out4321, out43, out432


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Edge_Module(nn.Module):
    def __init__(self, in_fea=[128, 256], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classifer = nn.Conv2d(mid_fea * 2, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 2)

    def forward(self, x2, x4):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge4], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge


class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.loact_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
            nn.Dropout(0.7),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        # self.afme = AFEM(channels)
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

        self.fuse = nn.Sequential(
        nn.Conv2d(2 * channels, channels, kernel_size=1), nn.BatchNorm2d(channels), nn.PReLU()
         )
    def forward(self, x, residual):
        xa = x + residual
        # xa = x * residual
        xg = self.global_att(xa)
        xl = self.loact_att(xa)
        # ca = self.channel_attention(xg + xl)
        x1 = xg * xl
        x2 = xg + xl
        xlg = self.fuse(torch.cat((x1,x2),1))

        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    unloader = torchvision.transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    net = COOPNet()
    a = torch.randn([2, 3, 384, 384])
    b = torch.randn([2, 3, 384, 384])
    s, e, s1 = net(a, b)
    print("s.shape:", e.shape)
