import torch
import numpy as np
import torch.nn as nn
import time
from skimage.segmentation import mark_boundaries

def convert_index(num_spixel_w=10,  max_spixels=100, feat_spixel_init=None):
    '''
    :param num_spixel_w: the number of spixels of an row
    :param max_spixels: the number of spixels
    :param feat_spixel_init:  1*1*H*W each pixel with corresponding spixel ids
    :return:
    '''
    if feat_spixel_init is not None:
        length = []
        ind_x = []
        ind_y = []
        feat_spixel_init = feat_spixel_init[0, 0]
        for i in range(max_spixels):
            id_y, id_x = np.where(feat_spixel_init==i)
            l = len(id_y)
            ind_y.extend(id_y.tolist())
            ind_x.extend(id_x.tolist())
            length.append(l)
        length = np.array(length)
        init_x = np.array(ind_x)
        init_y = np.array(ind_y)
        init_cum = np.cumsum(length)

        p2sp_index_, invisible = Passoc_Nspixel(feat_spixel_init, num_spixel_w, max_spixels)  # H*W*9, H*W*9
        length = []
        ind_x = []
        ind_y = []
        ind_z = []
        for i in range(max_spixels):
            id_y, id_x, id_z = np.where(p2sp_index_ == i)
            l = len(id_y)
            ind_y.extend(id_y)
            ind_x.extend(id_x)
            ind_z.extend(id_z)
            length.append(l)
        cir_x = np.array(ind_x)
        cir_y = np.array(ind_y)
        cir_z = np.array(ind_z)
        cir_cum = np.cumsum(length)

        return [init_x, init_y, init_cum], [cir_x, cir_y, cir_z, cir_cum], p2sp_index_, invisible

def SpixelFeature(feat, init_index, max_spixels=50, invisible_p=None):
    """
    init superpixel feature
    :param feat:  inputs feature of shape （B,C,H,W)
    :param init_index:   each spixel with corresponding pixel coordinates
    :param type: feature merge style default average
    :param max_spixels:  superpixel numbers
    :param ignore_id:
    :param rgb_scale:
    :param ignore_feat: 0
    :param invisible_p: ignore pixel of shape (B,H,W)
    :return: ave_feat: project pixel to superpixel feature; back_ave_feat: project superpixel to pixel feature
    """
    b, c, h, w = feat.shape
    init_x, init_y, init_l = init_index  #B*n  B*n  B*D  n=D*init_l[0,0]
    if len(init_x.shape) ==1:
        init_x = torch.from_numpy(init_x).unsqueeze(0)
        init_y = torch.from_numpy(init_y).unsqueeze(0)
        init_l = torch.from_numpy(init_l).unsqueeze(0)

    feat = feat[:, :, init_y[0], init_x[0]]  #B*C*n
    feat = feat.reshape(b, c, max_spixels, init_l[0, 0])
    # add ignore regions
    if invisible_p is not None:
        inv = invisible_p[:, init_y[0], init_x[0]]
        inv = inv.reshape(b, 1, max_spixels, init_l[0, 0])
        feat = (feat * (1 - inv)).sum(dim=3)
        valid = (1 - inv).sum(dim=3)
        ave_feat = feat / (valid + 1e-5)
    else:
        ave_feat = feat.sum(dim=3) / init_l[0, 0].float()

    return ave_feat

def Passoc_Nspixel(spixel_init, num_spixels_w, num_spixs):
  """
  calculate each pixel with corresponding 9 neighborhood spixel ids and whether is visible
  :param spixel_init:  (H,W) each pixel locates at which superpixel
  :param num_spixels_w:  the number of superpixel in one row
  :param num_spixs:  the number of superpixel in one superpixels
  :return: p2sp_index_: the index of spixel of a pixel H*W*9
            invisible: whether the surrounding spixel is available H*W*9
  """

  # b, c, h, w = pixel_features.shape
  center_spix_index = spixel_init[:, :]

  right_index = center_spix_index + 1
  left_index = center_spix_index - 1
  up_spix_index = center_spix_index - num_spixels_w
  up_right_index = up_spix_index + 1
  up_left_index = up_spix_index - 1
  down_spix_index = center_spix_index + num_spixels_w
  down_right_index = down_spix_index + 1
  down_left_index = down_spix_index - 1

  up_out_spix = up_spix_index <= -1
  down_out_spix = down_spix_index >= num_spixs
  right_out_spix = (center_spix_index + 1) % num_spixels_w == 0
  left_out_spix = center_spix_index % num_spixels_w == 0

  up_spix_index[up_out_spix] = center_spix_index[up_out_spix]
  down_spix_index[down_out_spix] = center_spix_index[down_out_spix]
  right_index[right_out_spix] = center_spix_index[right_out_spix]
  left_index[left_out_spix] = center_spix_index[left_out_spix]

  up_right_index[(right_out_spix + up_out_spix) > 0] = up_spix_index[(right_out_spix + up_out_spix) > 0]
  up_left_index[(left_out_spix + up_out_spix) > 0] = up_spix_index[(left_out_spix + up_out_spix) > 0]
  down_right_index[(right_out_spix + down_out_spix) > 0] = down_spix_index[(right_out_spix + down_out_spix) > 0]
  down_left_index[(left_out_spix + down_out_spix) > 0] = down_spix_index[(left_out_spix + down_out_spix) > 0]

  p2sp_index_ = np.stack([up_left_index, up_spix_index, up_right_index,
                          left_index, center_spix_index, right_index,
                          down_left_index, down_spix_index, down_right_index], axis=-1)  # H*W*9
  center_out_pixel = np.zeros_like(left_out_spix)


  invisible = np.stack(
      [(left_out_spix + up_out_spix) > 0, up_out_spix, (right_out_spix + up_out_spix) > 0,
       left_out_spix, center_out_pixel, right_out_spix,
       (left_out_spix + down_out_spix) > 0, down_out_spix, (right_out_spix + down_out_spix) > 0],
      axis=-1)

  return p2sp_index_, invisible

def Passoc(pixel_features, spixel_feat, p2sp_index_, invisible_, device, scale_value=-1):
    '''
    calculate the distance between pixel with surrounding 9 superpixel. each iteration spixel_init is fixed,
    only change the feature and association.
    :param pixel_features: (B,C,H,W)
    :param spixel_feat: (B,C,D) D is the number of surpixels
    :param p2sp_index_: B*H*W*9
    :param invisible_:  B*H*W*9
    :param scale_value:
    :return:
    '''
    b, c, h, w = pixel_features.shape
    # p2sp_index = p2sp_index_.reshape(1, h, w, 9).repeat(b, 1, 1, 1).long()
    if len(p2sp_index_.shape) == 3:
        p2sp_index_ = torch.from_numpy(p2sp_index_).unsqueeze(0)
        invisible_ = torch.from_numpy(invisible_).unsqueeze(0)

    p2sp_index = p2sp_index_.long()
    B_index = torch.arange(0, b).reshape(b, 1, 1, 1).repeat(1, h, w, 9).long().to(device)
    spixel_feat = spixel_feat.permute(0, 2, 1)  # B*C*D -> B*D*C
    p2sp_feat = spixel_feat[B_index, p2sp_index, :]  # B*H*W*9*C   (occupy storage 660M)
    p2sp_feat = p2sp_feat.permute(3, 0, 4, 1, 2)  # 9*B*C*H*W

    distance = torch.pow(p2sp_feat - pixel_features, 2.0)  # 9*B*C*H*W  (occupy storage 440M)
    distance = distance.sum(2).permute(1, 0, 2, 3)  # / c  # B*9*H*W

    invisible = invisible_.permute(0, 3, 1, 2).float()
    distance = distance * (1 - invisible) + 10000.0 * invisible
    #
    distance = distance * scale_value  # B*9*H*W
    return distance


def SpixelFeature2(pixel_features, pixel_assoc, cir_index, invisible, num_spixels_h, num_spixels_w):
    '''
    calculate spixel feature according to the similarity matrix between pixel and spixel
    :param pixel_features: B*C*H*W
    :param pixel_assoc:  B*9*H*W
    :param p2sp_index_: H*W*9
    :param invisible: H*W*9
    :param num_spixels_h:
    :param num_spixels_w:
    :return:
    '''

    b, c, h, w = pixel_features.shape
    num_spixels = num_spixels_w * num_spixels_h
    cir_x, cir_y, cir_z, cir_l = cir_index
    if len(cir_x.shape) ==1:
        cir_x = torch.from_numpy(cir_x).unsqueeze(0)
        cir_y = torch.from_numpy(cir_y).unsqueeze(0)
        cir_z = torch.from_numpy(cir_z).unsqueeze(0)
        cir_l = torch.from_numpy(cir_l).unsqueeze(0)
        invisible = torch.from_numpy(invisible).unsqueeze(0)

    feat = pixel_features[:, :, cir_y[0], cir_x[0]]  #B*C*n
    w = pixel_assoc[:, cir_z[0], cir_y[0], cir_x[0]].unsqueeze(1)  #B*1*n
    inv = invisible[:, cir_y[0], cir_x[0], cir_z[0]].unsqueeze(1)  #B*1*n

    s_feat = feat * w * (1 - inv.float())  #B*C*n
    weight = w * (1.0 - inv.float())  #B*1*n

    s_feat = s_feat.reshape(b, c, num_spixels, cir_l[0, 0])  #B*C*D*(n/D)
    weight = weight.reshape(b, 1, num_spixels, cir_l[0, 0])  #B*1*D*(n/D)

    weight = weight.sum(3)  #B*1*D
    s_feat = s_feat.sum(3)  #B*C*D

    S_feat = s_feat / (weight + 1e-5)
    S_feat = S_feat * (weight > 0.001).float()

    return S_feat


def compute_assignments(spixel_feat, pixel_features,
                        p2sp_index_, invisible, device):

    pixel_spixel_neg_dist = Passoc(pixel_features, spixel_feat, p2sp_index_, invisible, device)
    pixel_spixel_assoc = (pixel_spixel_neg_dist - pixel_spixel_neg_dist.max(1, keepdim=True)[0]).exp()
    pixel_spixel_assoc = pixel_spixel_assoc / (pixel_spixel_assoc.sum(1, keepdim=True))


    return pixel_spixel_assoc

def exec_iter(spixel_feat, trans_features, cir_index, p2sp_index_, invisible, num_spixels_h, num_spixels_w, device):

    # Compute pixel-superpixel assignments
    pixel_assoc = \
        compute_assignments(spixel_feat, trans_features, p2sp_index_, invisible, device)
    # t2 = time.time()
    spixel_feat1 = SpixelFeature2(trans_features, pixel_assoc, cir_index, invisible,
                                  num_spixels_h, num_spixels_w)
    # t3 = time.time()
    # print(f't2-t1:{t2-t1:.3f}, t3-t2:{t3-t2:.3f}')

    return spixel_feat1, pixel_assoc

def compute_final_spixel_labels(final_pixel_assoc, p2sp_index, num_spixels_h, num_spixels_w):
    """
    calculate the according spixel index of each pixel
    :param final_pixel_assoc: B*9*H*W
    :param p2sp_index: B*H*W*9 ndarray
    :param num_spixels_h:
    :param num_spixels_w:
    :return:
    """
    def RelToAbsIndex(rel_label, p2sp_index, num_spixels_h=1, num_spixels_w=1):
        """

        :param rel_label: B*H*W  the position(0-8) of the most similar spixel of  each pixel
        :param p2sp_index: B*H*W*9  ndarray
        :param num_spixels_h:
        :param num_spixels_w:
        :return: new_spix_indices : B*H*W each pixel corresponding to spixel index
        """
        b, h, w = rel_label.shape
        rel_label = rel_label.flatten(start_dim=1)  # b*n n=h*w
        if len(p2sp_index.shape)==3:
            p2sp_index = torch.from_numpy(p2sp_index).unsqueeze(0)

        p2sp_index = p2sp_index[0].flatten(end_dim=1)  # n*9
        index = torch.arange(end=h*w)
        index = index.reshape(1, h*w).repeat(b, 1)
        real_sindex = p2sp_index[index, rel_label]  # b*n
        real_sindex = real_sindex.reshape(b, h, w)

        return real_sindex

    rel_label = torch.argmax(final_pixel_assoc, 1)
    new_spix_indices = RelToAbsIndex(rel_label, p2sp_index)
    return new_spix_indices

def Semar(new_spixel_feat, new_spix_indices):
    """
    convert spixel feature to pixel via hard threshold
    :param new_spixel_feat:  iter results of size B*C*D
    :param new_spix_indices:  net final output of size B*H*W  each pixel corresponding to spixel index (hard decision)
    :return:
    """
    b, h, w = new_spix_indices.shape
    new_spixel_feat = new_spixel_feat.permute(0, 2, 1)  # B*D*C
    index = torch.arange(end=b)
    index = index.reshape(-1, 1, 1).repeat(1, h, w)
    feat = new_spixel_feat[index, new_spix_indices.long(), :]  # B*H*W*C
    feat_ = feat.permute(0, 3, 1, 2).contiguous()

    return feat_

def decode_features(pixel_spixel_assoc, spixel_feat, p2sp_index,
                    num_spixels_h, num_spixels_w, num_spixels, num_channels):
    """

    :param pixel_spixel_assoc: B*9*H*W the distance of each pixel and surrounding nine spixel
    :param spixel_feat: B*C*D spixel feature
    :param p2sp_index: B*H*W*9
    :param num_spixels_h:
    :param num_spixels_w:
    :param num_spixels:
    :param num_channels:
    :return:
    """
    b, _, h, w = pixel_spixel_assoc.shape
    _, c, d = spixel_feat.shape
    img_concat_spixel_feat = spixel_feat[:, :, p2sp_index[0].long()]  # B*C*H*W*9
    tiled_assoc = pixel_spixel_assoc.repeat(1, c, 1, 1)  # B*c9*H*W
    img_concat_spixel_feat = img_concat_spixel_feat.permute(0, 1, 4, 2, 3).reshape(b, -1, h, w)
    weighted_spixel_feat = img_concat_spixel_feat * tiled_assoc  # B*c9*H*W
    recon_feat = weighted_spixel_feat.reshape(b, c, 9, h, w)
    recon_feat = recon_feat.sum(2) + 1e-10  # B*c*H*W

    # norm
    try:
        assert recon_feat.min() >= 0., 'fails'
    except:
        import pdb
        pdb.set_trace()
    #
    recon_feat = recon_feat / recon_feat.sum(1, keepdim=True)


    return recon_feat


def get_spixel_image(given_img, spix_index):
    spixel_image = mark_boundaries(given_img / 255., spix_index.astype(int), color = (1,0,0))
    return spixel_image


if __name__ == '__main__':
    feat = torch.rand((2,5,50,50))
    feat_spixel_init = torch.from_numpy(np.random.randint(0, 50, [1,1,50,50]))
    p = SpixelFeature(feat, feat_spixel_init)
    s = 1