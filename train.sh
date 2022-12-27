python -m torch.distributed.launch 
    --nproc_per_node=4 --master_port=1234 main_train_psnr.py 
    --opt options/train_msrresnet_psnr.json  --dist True
# test