import functools
import torch
from torch.nn import init


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']


    # ----------------------------------------
    # denoising task
    # ----------------------------------------

    # ----------------------------------------
    # DnCNN
    # ----------------------------------------
    if net_type == 'dncnn':
        from models.network_dncnn import DnCNN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],  # total number of conv layers
                   act_mode=opt_net['act_mode'])

    # ----------------------------------------
    # Flexible DnCNN
    # ----------------------------------------
    elif net_type == 'fdncnn':
        from models.network_dncnn import FDnCNN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],  # total number of conv layers
                   act_mode=opt_net['act_mode'])

    # ----------------------------------------
    # FFDNet
    # ----------------------------------------
    elif net_type == 'ffdnet':
        from models.network_ffdnet import FFDNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'])

    # ----------------------------------------
    # others
    # ----------------------------------------

    # ----------------------------------------
    # super-resolution task
    # ----------------------------------------

    # ----------------------------------------
    # SRMD
    # ----------------------------------------
    elif net_type == 'srmd':
        from models.network_srmd import SRMD as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # super-resolver prior of DPSR
    # ----------------------------------------
    elif net_type == 'dpsr':
        from models.network_dpsr import MSRResNet_prior as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # modified SRResNet v0.0
    # ----------------------------------------
    elif net_type == 'msrresnet0':
        from models.network_msrresnet import MSRResNet0 as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # modified SRResNet v0.1
    # ----------------------------------------
    elif net_type == 'msrresnet1':
        from models.network_msrresnet import MSRResNet1 as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # RRDB
    # ----------------------------------------
    elif net_type == 'rrdb':  # RRDB
        from models.network_rrdb import RRDB as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # RRDBNet
    # ----------------------------------------
    elif net_type == 'rrdbnet':  # RRDBNet
        from models.network_rrdbnet import RRDBNet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nf=opt_net['nf'],
                   nb=opt_net['nb'],
                   gc=opt_net['gc'],
                   sf=opt_net['scale'])

    # ----------------------------------------
    # IMDB
    # ----------------------------------------
    elif net_type == 'imdn':  # IMDB
        from models.network_imdn import IMDN as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   upscale=opt_net['scale'],
                   act_mode=opt_net['act_mode'],
                   upsample_mode=opt_net['upsample_mode'])

    # ----------------------------------------
    # USRNet
    # ----------------------------------------
    elif net_type == 'usrnet':  # USRNet
        from models.network_usrnet import USRNet as net
        netG = net(n_iter=opt_net['n_iter'],
                   h_nc=opt_net['h_nc'],
                   in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'],
                   downsample_mode=opt_net['downsample_mode'],
                   upsample_mode=opt_net['upsample_mode']
                   )

    # ----------------------------------------
    # Deep Residual U-Net (drunet)
    # ----------------------------------------
    elif net_type == 'drunet':
        from models.network_unet import UNetRes as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   nb=opt_net['nb'],
                   act_mode=opt_net['act_mode'],
                   downsample_mode=opt_net['downsample_mode'],
                   upsample_mode=opt_net['upsample_mode'],
                   bias=opt_net['bias'])
    # ----------------------------------------
    # SCU-Net (scunet)
    # ----------------------------------------
    elif net_type == 'scunet':
        from models.network_scunet import SCUNet as net
        netG = net(in_nc=3)

    # ----------------------------------------
    # HAT
    # ----------------------------------------
    elif net_type == 'HAT':
        from models.network_HAT import HAT as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])

    # ----------------------------------------
    # XIXISR
    # ----------------------------------------
    elif net_type == 'XIXISR':
        from models.network_OURS import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge
    # ----------------------------------------
    elif net_type == 'in3con':
        from models.network_ours_in3con import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge upper阶段也用coor
    # ----------------------------------------
    elif net_type == 'in3con_merge_upper_coor':
        from models.network_ours_in3con_merge_upper_coor import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge
    # ----------------------------------------
    elif net_type == 'in3con_merge':
        from models.network_ours_in3con_merge import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge 但OCAB在小波域处理
    # ----------------------------------------
    elif net_type == 'in3con_merge_OCAB_wave':
        from models.network_ours_in3con_merge_OCAB_wave import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge 所有模块额外加hf
    # ----------------------------------------
    elif net_type == 'in3con_merge_all_hf':
        from models.network_ours_in3con_merge_all_hf import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge CAB Wave
    # ----------------------------------------
    elif net_type == 'in3con_merge_wave':
        from models.network_ours_in3con_merge_wave import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])    
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge  两个分支一个在小波域，一个在图像域
    # ----------------------------------------
    elif net_type == 'in3con_merge_bisapce':
        from models.network_ours_in3con_merge_bispace import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge 上采样后丢掉低频信息，和LR一起过HAG
    # ----------------------------------------
    elif net_type == 'in3con_merge_hf':
        from models.network_ours_in3con_merge_hf import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 比in3con_merge_hf多一步，在原图上就提取一次hf
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf':
        from models.network_ours_hf_in3con_merge_hf import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 比hf_in3con_merge_hf 只是在高频地方先把channel搞到和图像域相同的c
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_plus':
        from models.network_ours_hf_in3con_merge_hf_plus import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # hf_in3con_merge_hf 中的haar小波变成db小波
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_db':
        from models.network_ours_hf_in3con_merge_hf_db import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # hf_in3con_merge_hf 中的haar小波变成db小波
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_RRDB':
        from models.network_ours_hf_in3con_merge_hf_RRDB import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # hf_in3con_merge_hf + CAB->ECAB
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB':
        from models.network_ours_hf_in3con_merge_hf_ECAB import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # hf_in3con_merge_hf + CAB->ECAB + swin1.0->2.0
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB_v2':
        from models.network_ours_hf_in3con_merge_hf_ECAB_v2 import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # hf_in3con_merge_hf + CAB->ECAB + swin1.0->2.0
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB_mw':
        from models.network_ours_hf_in3con_merge_hf_ECAB_mw import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # ours  without cor
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB_woCor':
        from models.network_ours_hf_in3con_merge_hf_ECAB_woCor import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # ours  without wavelet
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB_woWave':
        from models.network_ours_hf_in3con_merge_hf_ECAB_woWave import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # ours  without Fourier
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB_woFor':
        from models.network_ours_hf_in3con_merge_hf_ECAB_woFor import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # ours  without ECAB
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB_woECA':
        from models.network_ours_hf_in3con_merge_hf_ECAB_woECA import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # ours  without Encoding
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB_woEncoding':
        from models.network_ours_hf_in3con_merge_hf_ECAB_woEncoding import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # in3_wave_respetive_con + CAB->ECAB 
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_ECAB_pro':
        from models.network_ours_hf_in3con_merge_hf_ECAB_pro import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 高低频分开处理 
    # ----------------------------------------
    elif net_type == 'in3_hf_LR_split_ECAB':
        from models.network_ours_in3_hf_split_ECAB import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 比in3con_merge_hf多一步，在原图上就提取一次hf without OCAB
    # ----------------------------------------
    elif net_type == 'hf_in3con_merge_hf_woOCAB':
        from models.network_ours_hf_in3con_merge_hf_woOCAB import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge 每个HAB里面都用额外提取hf一起，最后用waveup实现2倍超分
    # ----------------------------------------
    elif net_type == 'in3con_merge_hf_waveup':
        from models.network_ours_in3con_merge_hf_waveup import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # 先分别卷积，再con 再1*1 merge
    # ----------------------------------------
    elif net_type == 'in3con_merge_general':
        from models.network_ours_in3con_merge_general import XIXISR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # SwinIR
    # ----------------------------------------
    elif net_type == 'swinir':
        from models.network_swinir import SwinIR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])
    # ----------------------------------------
    # VRT
    # ----------------------------------------
    elif net_type == 'vrt':
        from models.network_vrt import VRT as net
        netG = net(upscale=opt_net['upscale'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   depths=opt_net['depths'],
                   indep_reconsts=opt_net['indep_reconsts'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   spynet_path=opt_net['spynet_path'],
                   pa_frames=opt_net['pa_frames'],
                   deformable_groups=opt_net['deformable_groups'],
                   nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'])

    # ----------------------------------------
    # others
    # ----------------------------------------
    # TODO

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # discriminator_vgg_96
    # ----------------------------------------
    if net_type == 'discriminator_vgg_96':
        from models.network_discriminator import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128':
        from models.network_discriminator import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_192
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_192':
        from models.network_discriminator import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128_SN
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128_SN':
        from models.network_discriminator import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'discriminator_patchgan':
        from models.network_discriminator import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'],
                             n_layers=opt_net['n_layers'],
                             norm_type=opt_net['norm_type'])

    elif net_type == 'discriminator_unet':
        from models.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
