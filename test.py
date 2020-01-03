import os
import torch
from torch.utils import data
from dataset import Dataset_T
from model import create_ssn_net, Loss
from PIL import Image
import scipy
from util import get_spixel_image
import sys
import numpy as np
import argparse
import imageio
import scipy.io as scio
# sys.path.append('')
from connectivity import enforce_connectivity
os.environ['CUDA_VISIBLE_DEVICES']='0'

def compute_spixels(num_spixel, num_steps, pre_model, out_folder):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        # os.makedirs(out_folder+'png')
        # os.makedirs(out_folder + 'mat')

    dtype = 'test'
    dataloader = data.DataLoader(Dataset_T(num_spixel=num_spixel),
                        batch_size=1, shuffle=False, num_workers=1)
    model = create_ssn_net(num_spixels=num_spixel, num_iter=num_steps, num_spixels_h=10, num_spixels_w=10, dtype=dtype, ssn=0)
    model = torch.nn.DataParallel(model)
    if pre_model is not None:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(pre_model))
        else:
           model.load_state_dict(torch.load(pre_model, map_location='cpu'))
    else:
        raise ('no model')
    criten = Loss()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    for iter, [inputs, num_h, num_w, init_index, cir_index, p2sp_index_, invisible, file_name] in enumerate(dataloader):
        with torch.no_grad():
            img = inputs['img'].to(device)
            label = inputs['label'].to(device)
            problabel = inputs['problabel'].to(device)
            num_h = num_h.to(device)
            num_w = num_w.to(device)
            init_index = [x.to(device) for x in init_index]
            cir_index = [x.to(device) for x in cir_index]
            p2sp_index_ = p2sp_index_.to(device)
            invisible = invisible.to(device)
            recon_feat2, recon_label, new_spix_indices = model(img, p2sp_index_, invisible, init_index, cir_index, problabel, num_h,
                                             num_w, device)
            # loss, loss_1, loss_2 = criten(recon_feat2, img, recon_label, label)

            given_img = np.asarray(Image.open(file_name[0]))
            h, w = given_img.shape[0], given_img.shape[1]
            new_spix_indices = new_spix_indices[:, :h, :w].contiguous()
            spix_index = new_spix_indices.cpu().numpy()[0]
            spix_index = spix_index.astype(int)

            if enforce_connectivity:
                segment_size = (given_img.shape[0] * given_img.shape[1]) / (int(num_h*num_w) * 1.0)
                min_size = int(0.06 * segment_size)
                max_size = int(3 * segment_size)
                spix_index = enforce_connectivity(spix_index[np.newaxis, :, :], min_size, max_size)[0]
            # given_img_ = np.zeros([spix_index.shape[0], spix_index.shape[1], 3], dtype=np.int)
            # h, w = given_img.shape[0], given_img.shape[1]
            # given_img_[:h, :w] = given_img

            counter_image = np.zeros_like(given_img)
            counter_image = get_spixel_image(counter_image, spix_index)
            spixel_image = get_spixel_image(given_img, spix_index)

            imgname = file_name[0].split('/')[-1][:-4]
            out_img_file = out_folder + imgname + '_bdry_.jpg'
            imageio.imwrite(out_img_file, spixel_image)
            # out_file = out_folder + imgname + '.npy'
            # np.save(out_file, spix_index)

            # validation code only for sp_pix 400
            # out_file_mat = out_folder + 'mat/'+ imgname + '.mat'
            # scio.savemat(out_file_mat, {'segs': spix_index})

            # out_count_file = out_folder + 'png/' + imgname + '.png'
            # imageio.imwrite(out_count_file, counter_image)
            print(iter)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_spixels', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--result_dir', type=str, default='./results/pix/')
    parser.add_argument('--pre_dir', type=str, default='./45000_0.527_model.pt')

    var_args = parser.parse_args()
    compute_spixels(var_args.n_spixels, var_args.num_steps,
                    var_args.pre_dir, var_args.result_dir)





