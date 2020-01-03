import torch
import torch.nn as nn
import numpy as np
import pdb
class position_color_loss(nn.Module):
    def __init__(self, pos_weight=0.4, col_weight=0.):
        """
        :param pos_weight:
        :param col_weight:
        """
        super(position_color_loss, self).__init__()
        self.pos_weight = pos_weight
        self.col_weight = col_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, recon_feat, pixel_features):
        """

        :param recon_feat: B*C*H*W restructure  pixel feature (c=RGBplusXY)
        :param pixel_features: B*C*H*W original pixel feature
        :return:
        """
        # pdb.set_trace()
        pos_recon_feat = recon_feat[:, :2, :, :]
        color_recon_feat = recon_feat[:, 2:, :, :]
        pos_pix_feat = pixel_features[:, :2, :, :]
        color_pix_feat = pixel_features[:, 2:, :, :]

        pos_loss = self.mse_loss(pos_recon_feat, pos_pix_feat)
        color_loss = self.mse_loss(color_recon_feat, color_pix_feat)

        pos_clor_loss = pos_loss * self.pos_weight + color_loss * self.col_weight

        return pos_clor_loss

class LossWithoutSoftmax(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_label=255):
        super(LossWithoutSoftmax, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.NLLloss = nn.NLLLoss(reduction='none')
    def forward(self, recon_label3, label, invisible_p=None):
        """

        :param recon_label3: B*C*H*W  reconstructure label by soft threshold
        :param label:  B*1*H*W gt label
        :param invisible_p: B*H*W invisible pixel (ignore region)
        :return:
        """
        # pdb.set_trace()
        label = label[:, 0, ...]

        # add ignore region
        if invisible_p is not None:
            ignore = invisible_p == 1.
        elif self.ignore_label is not None:
            ignore = label == self.ignore_label
        else:
            raise IOError
        label[ignore] = 0

        loss = self.NLLloss(recon_label3, label)  # B*H*W
        #
        # view_loss = loss.data.numpy()
        #
        loss = -1 * loss[1 - ignore]
        loss = -1 * torch.log(loss)
        loss = loss.mean() * self.loss_weight

        return loss


