import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import cv2

class DatasetPlain(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetPlain, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.upscale = opt['scale'] 
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64

        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_H'])

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        # img_L = util.imread_uint(L_path, self.n_channels)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            scale = random.uniform(1.0, 5.0) #下采样的倍数
            # scale = 1.0
            rnd_h = random.randint(0, max(0, H - int(self.patch_size * scale)))
            rnd_w = random.randint(0, max(0, W - int(self.patch_size * scale)))
            # print(img_H.shape)
            # assert(0==1)
            # patch_L = img_L[rnd_h:rnd_h + int(self.patch_size * scale), rnd_w:rnd_w + int(self.patch_size*scale), :]
            patch_H = img_H[rnd_h:rnd_h + int(self.patch_size * scale), rnd_w:rnd_w + int(self.patch_size*scale), :]
            patch_H = util.imresize_np(patch_H, 1 / scale, True)
            patch_L = util.imresize_np(patch_H, 1 / self.upscale, True)


            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)


        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            scale = 1.0
            img_L = util.imresize_np(img_H, 1 / self.upscale, True)
            # img_L = cv2.resize(img_H,(int(img_H.shape[0]/self.upscale),int(img_H.shape[1]/self.upscale)), interpolation = cv2.INTER_CUBIC)
            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)
        
        if 'postdam' in H_path:
            resolution = 0.05
        elif 'JAX' in H_path or 'OMA' in H_path:
            resolution = 0.35
        else:
            resolution = 0.3
        # resolution = 1.0
        
        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path,'scale': scale, 'resolution': resolution }

    def __len__(self):
        return len(self.paths_H)
