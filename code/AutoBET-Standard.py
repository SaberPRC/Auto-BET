import os
import ants
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import SimpleITK as sitk
from tqdm import tqdm
from IPython import embed
from skimage import measure
from itertools import product
import torch.nn.functional as F
from scipy import ndimage


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm3d(self.half, affine=True)
        self.BN = nn.BatchNorm3d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class BasicBlock(nn.Module):
    # TODO: basic convolutional block, conv -> batchnorm -> activate
    def __init__(self, in_channels, out_channels, kernel_size, padding, activate=True, norm='IBNa', act='LeakyReLU'):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, bias=True)

        if norm == 'IBNa':
            self.bn = IBN(out_channels)
        else:
            self.bn = nn.BatchNorm3d(out_channels)

        if act == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.activate = nn.LeakyReLU(0.2)

        self.en_activate = activate

    def forward(self, x):
        if self.en_activate:
            return self.activate(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    # TODO: basic residual block established by BasicBlock
    def __init__(self, in_channels, out_channels, kernel_size, padding, nums, norm='IBNa',act='LeakyReLU'):
        '''
        TODO: initial parameters for basic residual network
        :param in_channels: input channel numbers
        :param out_channels: output channel numbers
        :param kernel_size: convoluition kernel size
        :param padding: padding size
        :param nums: number of basic convolutional layer
        '''
        super(ResidualBlock, self).__init__()

        layers = list()

        self.norm = norm

        for _ in range(nums):
            if _ != nums - 1:
                layers.append(BasicBlock(in_channels, out_channels, kernel_size, padding, True, norm, act))
            else:
                layers.append(BasicBlock(in_channels, out_channels, kernel_size, padding, False, None, act))

        self.do = nn.Sequential(*layers)

        if act == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.activate = nn.LeakyReLU(0.2)

        self.IN = nn.InstanceNorm3d(out_channels, affine=True) if norm == 'IBNb' else None

    def forward(self, x):
        output = self.do(x)
        if self.IN is not None:
            return self.activate(self.IN(output + x))
        else:
            return self.activate(output + x)


class InputTransition(nn.Module):
    # TODO: input transition convert image to feature space
    def __init__(self, in_channels, out_channels, norm=None):
        '''
        TODO: initial parameter for input transition <input size equals to output feature size>
        :param in_channels: input image channels
        :param out_channels: output feature channles
        '''
        super(InputTransition, self).__init__()
        self.norm = norm
        self.trans = BasicBlock(in_channels, out_channels, 3, 1, True, norm, 'LeakyReLU')

    def forward(self, x):
        out = self.trans(x)
        return out


class OutputTransition(nn.Module):
    # TODO: feature map convert to predict results
    def __init__(self, in_channels, out_channels, act='sigmoid'):
        '''
        TODO: initial for output transition
        :param in_channels: input feature channels
        :param out_channels: output results channels
        :param act: final activate layer sigmoid or softmax
        '''
        super(OutputTransition, self).__init__()
        assert act == 'sigmoid' or act =='softmax', \
            'final activate layer should be sigmoid or softmax, current activate is :{}'.format(act)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.act = act

    def forward(self, x):
        out = self.activate1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        if self.act == 'sigmoid':
            return self.sigmoid(out)
        elif self.act == 'softmax':
            return self.softmax(out)


class DownTransition(nn.Module):
    # TODO: fundamental down-sample layer <inchannel -> 2*inchannel>
    def __init__(self, in_channels, nums, norm=None, act='LeakyReLU'):
        '''
        TODO: intial for down-sample
        :param in_channels: inpuit channels
        :param nums: number of reisidual block
        '''
        super(DownTransition, self).__init__()

        out_channels = in_channels * 2
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums, norm, act)

    def forward(self, x):
        out = self.activate1(self.bn1(self.down(x)))
        out = self.residual(out)
        return out


class UpTransition(nn.Module):
    # TODO: fundamental up-sample layer (inchannels -> inchannels/2)
    def __init__(self, in_channels, out_channels, nums):
        '''
        TODO: initial for up-sample
        :param in_channels: input channels
        :param out_channels: output channels
        :param nums: number of residual block
        '''
        super(UpTransition, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels//2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels//2)
        self.activate = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums)

    def forward(self, x, skip_x):
        out = self.up(x)
        out = self.activate(self.bn(self.conv1(out)))
        out = torch.cat((out,skip_x), 1)
        out = self.residual(out)

        return out


class SegNetMultiScale(nn.Module):
    # TODO: fundamental segmentation framework
    # Multi-Scale strategy using different crop size and normalize to same size
    def __init__(self, in_channels, out_channels, norm=None):
        super().__init__()
        self.in_tr_s = InputTransition(in_channels, out_channels=16)
        self.in_tr_b = InputTransition(in_channels, out_channels=16)

        self.fuse = BasicBlock(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        self.down_32 = DownTransition(16, 2)
        self.down_64 = DownTransition(32, 2)
        self.down_128 = DownTransition(64, 2)
        self.down_256 = DownTransition(128, 4)

        self.bottleneck = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, padding=1, nums=4)

        self.up_256 = UpTransition(256, 256, 4)
        self.up_128 = UpTransition(256, 128, 2)
        self.up_64 = UpTransition(128, 64, 2)
        self.up_32 = UpTransition(64, 32, 2)

        self.out_tr = OutputTransition(32, out_channels, 'softmax')
    def forward(self, x):
        B, C, W, H, D = x.shape
        B_s, C_s, W_s, H_s, D_s = B, C, W - 32, H - 32, D - 32

        x_s = x[:, :, 16:W - 16, 16:H - 16, 16:D - 16]
        x_b = F.interpolate(x, size=[W_s, H_s, D_s])

        out_16_s = self.in_tr_s(x_s)
        out_16_b = self.in_tr_b(x_b)

        out_16 = torch.cat([out_16_s, out_16_b], dim=1)

        out16 = self.fuse(out_16)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        out256 = self.bottleneck(out256)

        out = self.up_256(out256, out128)
        out = self.up_128(out, out64)
        out = self.up_64(out, out32)
        out = self.up_32(out, out16)

        out = self.out_tr(out)

        return out


def _model_init(args, model_path):
    # initialize model and load pretrained model parameters
    model = SegNetMultiScale(args.num_modalities, args.num_classes+1)
    model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def _seg_to_label(seg):
    '''
    TODO: Labeling each single annotation in one image (single label to multiple label)
    :param seg: single label annotation (numpy data)
    :return: multiple label annotation
    '''
    labels, num = measure.label(seg, return_num=True)
    labels = labels.astype(np.float32)
    return labels, num


def _select_top_k_region(img, k=2):
    '''
    TODO: Functions to select top k connection regions
    :param img: numpy array with multiple regions
    :param k: number of selected regions
    :return: selected top k region data
    '''
    # seg to labels
    labels, nums = _seg_to_label(img)
    rec = list()

    for idx in range(1, nums+1):
        subIdx = np.where(labels==idx)
        rec.append(len(subIdx[0]))
    rec_sort = rec.copy()
    rec_sort.sort()

    rec = np.array(rec)
    index = np.where(rec >= rec_sort[-k])[0]
    index = list(index)

    for idx in index:
        labels[labels==idx+1] = 1000000

    labels[labels != 1000000] = 0
    labels[labels == 1000000] = 1

    return labels


def _ants_img_info(img_path):
    img = ants.image_read(img_path)
    return img.origin, img.spacing, img.direction, img.numpy()


def _normalize_z_score(data, clip=True):
    '''
    funtions to normalize data to standard distribution using (data - data.mean()) / data.std()
    :param data: numpy array
    :param clip: whether using upper and lower clip
    :return: normalized data by using z-score
    '''
    if clip == True:
        bounds = np.percentile(data, q=[0.001, 99.999])
        data[data <= bounds[0]] = bounds[0]
        data[data >= bounds[1]] = bounds[1]

    return (((data - data.min()) / (data.max() - data.min())) - 0.5) * 2


def calculate_patch_index(target_size, patch_size, overlap_ratio = 0.25):
    shape = target_size

    gap = int(patch_size[0] * (1-overlap_ratio))
    index1 = [f for f in range(shape[0])]
    index_x = index1[::gap]
    index2 = [f for f in range(shape[1])]
    index_y = index2[::gap]
    index3 = [f for f in range(shape[2])]
    index_z = index3[::gap]

    index_x = [f for f in index_x if f < shape[0] - patch_size[0]]
    index_x.append(shape[0]-patch_size[0])
    index_y = [f for f in index_y if f < shape[1] - patch_size[1]]
    index_y.append(shape[1]-patch_size[1])
    index_z = [f for f in index_z if f < shape[2] - patch_size[2]]
    index_z.append(shape[2]-patch_size[2])

    start_pos = list()
    loop_val = [index_x, index_y, index_z]
    for i in product(*loop_val):
        start_pos.append(i)
    return start_pos


def _get_pred(args, model, img):
    if len(img.shape) == 4:
        img = torch.unsqueeze(img, dim=0)
    m = nn.ConstantPad3d(16, 0)
    B, C, W, H, D = img.shape
    pos = calculate_patch_index((W, H, D), args.crop_size, args.overlap_ratio)

    pred_rec_s = torch.zeros((args.num_classes+1, W, H, D))
    freq_rec = torch.zeros((args.num_classes+1, W, H, D))

    for start_pos in pos:
        patch = img[:,:,start_pos[0]:start_pos[0]+args.crop_size[0], start_pos[1]:start_pos[1]+args.crop_size[1], start_pos[2]:start_pos[2]+args.crop_size[2]]

        model_out_s = model(patch)
        model_out_s = m(model_out_s)
        model_out_s = model_out_s.cpu().detach()

        pred_rec_s[:, start_pos[0]:start_pos[0]+args.crop_size[0], start_pos[1]:start_pos[1]+args.crop_size[1], start_pos[2]:start_pos[2]+args.crop_size[2]] += model_out_s[0,:,:,:,:]
        freq_rec[:, start_pos[0]:start_pos[0]+args.crop_size[0], start_pos[1]:start_pos[1]+args.crop_size[1], start_pos[2]:start_pos[2]+args.crop_size[2]] += 1

    pred_rec_s = pred_rec_s / freq_rec
    pred_rec_s = pred_rec_s[:, 16:W-16, 16:H-16, 16:D-16]

    return pred_rec_s


def get_pred(args, model, img_path1):
    origin, spacing, direction, img = _ants_img_info(img_path1)
    img = _normalize_z_score(img)
    img = np.pad(img, ((16, 16), (16, 16), (16, 16)), 'constant')
    img = torch.from_numpy(img).type(torch.float32)
    img = img.to(args.device)
    img = img.unsqueeze(0)

    pred = _get_pred(args, model, img)
    pred = pred.argmax(0)
    pred = pred.numpy().astype(np.float32)

    pred = _select_top_k_region(pred, 1)
    pred = pred.astype(np.float32)
    pred = ndimage.binary_fill_holes(pred).astype(np.float32)

    ants_img_pred_seg = ants.from_numpy(pred, origin, spacing, direction)

    return ants_img_pred_seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infant segmentation Experiment settings')
    parser.add_argument('--num_classes', type=int, default=1, help='number of output channels')
    parser.add_argument('--num_modalities', type=int, default=1, help='number of input channels')
    parser.add_argument('--model_path', type=str, default='/path/to/pretrained/Standard/AutoBET/Model', help='Pretrained model path')
    parser.add_argument('--device', type=str, default='cuda', help='specify device type: cuda or cpu?')
    parser.add_argument('--crop_size', type=tuple, default=(160, 160, 160), help='patch size')
    parser.add_argument('--overlap_ratio', type=float, default=0.5, help='Overlap ratio to extract '
                                                                          'patches for single image inference')
    parser.add_argument('--input', type=str, default='/path/to/persudo/brain/image', help='None')
    parser.add_argument('--output_brain', type=str, default='/path/to/save/brain', help='None')
    parser.add_argument('--output_brain_mask', type=str, default='/path/to/save/brain/mask', help='None')

    args = parser.parse_args()

    model = _model_init(args, args.model_path)

    pred = get_pred(args, model, arg.input)
    ants.image_write(pred, args.output_brain_mask)

    pred = pred.numpy()
    origin, spacing, direction, img = _ants_img_info(arg.input)
    pred_brain = img * pred
    pred_brain = ants.from_numpy(pred_brain, origin, spacing, direction)

    ants.image_write(pred_brain, args.output_brain)

