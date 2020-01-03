from torch.utils import data
import os
import scipy
from scipy.io import loadmat
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage import io
import numpy as np
from random import Random
from scipy import interpolate
from util import convert_index
RAND_SEED = 2356
myrandom = Random(RAND_SEED)

def convert_label(label, num=50):

    problabel = np.zeros((1, num, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= num:
            print(np.unique(label).shape)
            break
            # raise IOError
        else:
            problabel[:, ct, :, :] = (label == t)
        ct = ct + 1

    label2 = np.squeeze(np.argmax(problabel, axis = 1))

    return label2, problabel

def transform_and_get_image(im, max_spixels, out_size):

    height = im.shape[0]
    width = im.shape[1]

    out_height = out_size[0]
    out_width = out_size[1]

    pad_height = out_height - height
    pad_width = out_width - width
    im = np.lib.pad(im, ((0, pad_height), (0, pad_width), (0, 0)), 'constant',
                    constant_values=-10)
    im = np.expand_dims(im, axis=0)
    return im

def get_spixel_init(num_spixels, img_width, img_height):
    """
    :return each pixel belongs to which pixel
    """

    k = num_spixels
    k_w = int(np.floor(np.sqrt(k * img_width / img_height)))
    k_h = int(np.floor(np.sqrt(k * img_height / img_width)))

    spixel_height = img_height / (1. * k_h)
    spixel_width = img_width / (1. * k_w)

    h_coords = np.arange(-spixel_height / 2. - 1, img_height + spixel_height - 1,
                         spixel_height)
    w_coords = np.arange(-spixel_width / 2. - 1, img_width + spixel_width - 1,
                         spixel_width)
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    spix_values = np.pad(spix_values, 1, 'symmetric')
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing = 'ij'))
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()

    spixel_initmap = f(all_points).reshape((img_height,img_width))

    feat_spixel_initmap = spixel_initmap
    return [spixel_initmap, feat_spixel_initmap, k_w, k_h]

def transform_and_get_spixel_init(max_spixels, out_size):

    out_height = out_size[0]
    out_width = out_size[1]

    spixel_init, feat_spixel_initmap, k_w, k_h = \
        get_spixel_init(max_spixels, out_width, out_height)
    spixel_init = spixel_init[None, None, :, :]
    feat_spixel_initmap = feat_spixel_initmap[None, None, :, :]

    return spixel_init, feat_spixel_initmap, k_h, k_w
def get_rand_scale_factor():

    rand_factor = np.random.normal(1, 0.75)

    s_factor = np.min((3.0, rand_factor))
    s_factor = np.max((0.75, s_factor))

    return s_factor
def scale_image(im, s_factor):

    s_img = scipy.ndimage.zoom(im, (s_factor, s_factor, 1), order = 1)

    return s_img
def scale_label(label, s_factor):

    s_label = scipy.ndimage.zoom(label, (s_factor, s_factor), order = 0)

    return s_label

def PixelFeature(img, color_scale=None, pos_scale=None, type=None):
    b,h,w,c = img.shape
    feat = img * color_scale
    if type is 'RGB_AND_POSITION':  #yxrcb
        x_axis = np.arange(0, w, 1)
        y_axis = np.arange(0, h, 1)
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
        yx = np.stack([y_mesh, x_mesh], axis=-1)
        yx_scaled = yx * pos_scale
        yx_scaled = np.repeat(yx_scaled[np.newaxis], b, axis=0)
        feat = np.concatenate([yx_scaled, feat], axis=-1)
    return feat

class Dataset(data.Dataset):
    def __init__(self, num_spixel, root=None, patch_size=None, dtype='train'):
        self.patch_size = patch_size
        # self.width = width
        self.num_spixel = num_spixel
        self.out_types = ['img', 'spixel_init', 'feat_spixel_init', 'label', 'problabel']

        self.root = root
        self.dtype = dtype
        self.data_dir = os.path.join(self.root, 'BSR', 'BSDS500', 'data')

        self.split_list = open(os.path.join(root, dtype + '.txt')).readlines()
        self.img_dir = os.path.join(self.data_dir, 'images', self.dtype)
        self.gt_dir = os.path.join(self.data_dir, 'groundTruth', self.dtype)

        # init pixel-spixel index
        self.out_spixel_init, self.feat_spixel_init, self.spixels_h, self.spixels_w = \
            transform_and_get_spixel_init(self.num_spixel, [patch_size[0], patch_size[1]])
        self.init, self.cir, self.p2sp_index_, self.invisible = convert_index(self.spixels_w, self.spixels_w*self.spixels_h, self.feat_spixel_init)
        self.invisible = self.invisible.astype(np.float)


    def __getitem__(self, item):
        img_name = self.split_list[item].rstrip('\n')
        e=io.imread(os.path.join(self.img_dir, img_name + '.jpg'))
        image = img_as_float(io.imread(os.path.join(self.img_dir, img_name + '.jpg')))
        s_factor = get_rand_scale_factor()
        image = scale_image(image, s_factor)
        im = rgb2lab(image)
        h, w, _ = im.shape

        gtseg_all = loadmat(os.path.join(self.gt_dir, img_name + '.mat'))
        t = np.random.randint(0, len(gtseg_all['groundTruth'][0]))
        gtseg = gtseg_all['groundTruth'][0][t][0][0][0]
        gtseg = scale_label(gtseg, s_factor)

        if np.random.uniform(0, 1) > 0.5:
            im = im[:, ::-1, ...]
            gtseg = gtseg[:, ::-1]

        if self.patch_size == None:
            raise ('not define the output size')
        else:
            out_height = self.patch_size[0]
            out_width = self.patch_size[1]

        if out_height > h:
            raise ("Patch size is greater than image size")

        if out_width > w:
            raise ("Patch size is greater than image size")

        start_row = myrandom.randint(0, h - out_height)
        start_col = myrandom.randint(0, w - out_width)
        im_cropped = im[start_row: start_row + out_height,
                     start_col: start_col + out_width, :]
        out_img = transform_and_get_image(im_cropped, self.num_spixel, [out_height, out_width])
        # add xy information
        out_img = PixelFeature(out_img, color_scale=0.26, pos_scale=0.125, type='RGB_AND_POSITION')

        gtseg_cropped = gtseg[start_row: start_row + out_height,
                        start_col: start_col + out_width]
        label_cropped, problabel_cropped = convert_label(gtseg_cropped)

        inputs = {}
        for in_name in self.out_types:
            if in_name == 'img':
                inputs['img'] = np.transpose(out_img[0], [2, 0, 1]).astype(np.float32)
            if in_name == 'spixel_init':
                inputs['spixel_init'] = self.out_spixel_init[0].astype(np.float32)
            if in_name == 'feat_spixel_init':
                inputs['feat_spixel_init'] = self.feat_spixel_init[0].astype(np.float32)
            if in_name == 'label':
                label_cropped = np.expand_dims(np.expand_dims(label_cropped, axis=0), axis=0)
                inputs['label'] = label_cropped[0]
            if in_name == 'problabel':
                inputs['problabel'] = problabel_cropped[0]

        return inputs, self.spixels_h, self.spixels_w, self.init, self.cir, self.p2sp_index_, self.invisible

    def __len__(self):
        return len(self.split_list)

class Dataset_T(data.Dataset):
    def __init__(self, num_spixel, root='', patch_size=None, dtype='test'):
        self.patch_size = patch_size
        self.num_spixel = num_spixel
        self.out_types = ['img', 'spixel_init', 'feat_spixel_init', 'label', 'problabel']

        self.root = root
        self.dtype = dtype
        self.data_dir = os.path.join(self.root, 'BSR', 'BSDS500', 'data')

        self.split_list = open(os.path.join(root, dtype + '.txt')).readlines()
        self.img_dir = os.path.join(self.data_dir, 'images', self.dtype)
        self.gt_dir = os.path.join(self.data_dir, 'groundTruth', self.dtype)

    def __getitem__(self, item):
        img_name = self.split_list[item].rstrip('\n')
        image = img_as_float(io.imread(os.path.join(self.img_dir, img_name + '.jpg')))

        im = rgb2lab(image)
        h, w, _ = im.shape

        gtseg_all = loadmat(os.path.join(self.gt_dir, img_name + '.mat'))
        t = 0 #np.random.randint(0, len(gtseg_all['groundTruth'][0]))
        gtseg = gtseg_all['groundTruth'][0][t][0][0][0]

        k = self.num_spixel
        k_w = int(np.floor(np.sqrt(k * w / h)))
        k_h = int(np.floor(np.sqrt(k * h / w)))
        spixel_height = h / (1. * k_h)
        spixel_width = w / (1. * k_w)

        out_height = int(np.ceil(spixel_height) * k_h)
        out_width = int(np.ceil(spixel_width) * k_w)

        out_img = transform_and_get_image(im, self.num_spixel, [out_height, out_width])
        # add xy information
        pos_scale = 2.5 * max(k_h/out_height, k_w/out_width)
        out_img = PixelFeature(out_img, color_scale=0.26, pos_scale=pos_scale, type='RGB_AND_POSITION')

        gtseg_ = np.ones_like(out_img[0, :, :, 0]) * 49
        gtseg_[:h, :w] = gtseg
        label_cropped, problabel_cropped = convert_label(gtseg_)

        self.out_spixel_init, self.feat_spixel_init, self.spixels_h, self.spixels_w = \
            transform_and_get_spixel_init(self.num_spixel, [out_height, out_width])
        self.init, self.cir, self.p2sp_index_, self.invisible = convert_index(self.spixels_w,
                                                                              self.spixels_w * self.spixels_h,
                                                                              self.feat_spixel_init)
        self.invisible = self.invisible.astype(np.float)

        inputs = {}
        for in_name in self.out_types:
            if in_name == 'img':
                inputs['img'] = np.transpose(out_img[0], [2, 0, 1]).astype(np.float32)
            if in_name == 'spixel_init':
                inputs['spixel_init'] = self.out_spixel_init[0].astype(np.float32)
            if in_name == 'feat_spixel_init':
                inputs['feat_spixel_init'] = self.feat_spixel_init[0].astype(np.float32)
            if in_name == 'label':
                label_cropped = np.expand_dims(np.expand_dims(label_cropped, axis=0), axis=0)
                inputs['label'] = label_cropped[0]
            if in_name == 'problabel':
                inputs['problabel'] = problabel_cropped[0]

        return inputs, self.spixels_h, self.spixels_w, self.init, self.cir, self.p2sp_index_, self.invisible, \
               os.path.join(self.img_dir, img_name + '.jpg')

    def __len__(self):
        return len(self.split_list)

if __name__ == '__main__':
    data = Dataset(100, patch_size=[200, 200])
    for i in data:
        s=1


