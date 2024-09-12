import os
import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import hydra
from tqdm import tqdm

from src.models import SetCriterion
from src.datasets import collateFunction, COCODataset
from src.utils import load_model
from src.utils.misc import cast2Float
from src.utils.utils import load_weights
from src.utils.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU
from src.models.matcher import HungarianMatcher


import torch
from torch import nn

from modules import PixelShuffle,ResidualBlocksWithInputConv,flow_warp


class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)]
        )

        #self.load_state_dict(torch.load('spynet_20210409-c6c1bd09.pth'))

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Note that in this function, the images are already resized to a
        multiple of 32.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False
                )
            )
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False
                )
            )
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.
        This function computes the optical flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].
        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


class basicVSR(nn.Module):
    def __init__(self,scale_factor=4, mid_channels=64, num_blocks=30, spynet_pretrained=None):
        super().__init__()
        self.scale_factor=scale_factor
        self.mid_channels = mid_channels

        #alignment(optical flow network)
        self.spynet = self.get_spynet(spynet_pretrained)
        
        #propagation
        self.backward_resblocks=ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks=ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)

        #upsample
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShuffle(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShuffle(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_spynet(self,pretrained):
        model=SPyNet()
        if pretrained:
            model_p=model.state_dict()
            pre_p=torch.load(pretrained)
            ppl=list(pre_p)
            for i,k in enumerate(model_p.keys()):
                if i<2:
                    continue
                model_p[k]=pre_p[ppl[i-2]]
            model.load_state_dict(model_p)
        return model
    
    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True
    
    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.
        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, lrs):
        """Forward function for BasicVSR.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).(if scale_factor=4)
        """

        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs)

        # backward-time propgation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = torch.cat([outputs[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.img_upsample(lr_curr)
            out += base
            outputs[i] = out

        return torch.stack(outputs, dim=1)



@hydra.main(config_path="../config", config_name="config")
def main(args):
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outputDir, exist_ok=True)

    matcher = HungarianMatcher(args.classCost, args.bboxCost, args.giouCost)

    # load data and model
    test_dataset = COCODataset(args.dataDir, args.testAnnFile, args.numClass, args.sequenceLength, args.scaleMetadata)

    # create a folder using args.model param in the outputDir
    fileDir = os.path.join(args.outputDir, args.model)
    print('fileDir', fileDir)
    os.makedirs(fileDir, exist_ok=True)

    # load the model
    model = basicVSR()
    
    model.eval()
    with torch.no_grad():
        for i in range(10,15):

            

            img, metadata, target = test_dataset.__getitem__(i)  

            # add two extra dimensions to the tensor, one for the batch size and one for the number of frames
            img = img.unsqueeze(0).unsqueeze(0)
            

            previous_frame = img    

            for j in range(5):
                tmp, metadata, target = test_dataset.__getitem__(i-j)
                previous_frame = torch.cat((previous_frame, tmp.unsqueeze(0).unsqueeze(0)), 1)

            out = model(previous_frame)
            print('out', out.shape)



            
            

            
            


   
if __name__ == '__main__':
    main()