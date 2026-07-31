# 調整sigma_offsets, 讓LL偏fidelity, 高頻偏realism, 並且讓四個子帶均等起步
    # wd_fn = VGG16WaveletWassersteinDistortion(
    #     num_levels=5, dwt_levels=1,
    #     learnable_weights=False,          # 关掉,别让它自己推向 HH
    #     sigma_offsets=(-0.5, 0.0, 0.0, 0.0),  # HH 从 1.0 降到 0.5,别过度realism
    #     ll_weight_boost=0.3,              # LL 温和回补一点(不是之前的1.0)
    # ).to(DEVICE)
name="dcae_wd32_512_static_from0240_mse_20_tuning_mask_msd02"
python train_mask_msd.py -d /home/at9529/ycw.cs14/Michael/MLIC/mlic_train_100k --cuda --checkpoint /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/2.4checkpoint_best.pth.tar \
    --lambda 1 --wd-weight 0.21 --mse-weight 0 --final-mse-weight 20 \
    --skip-warmup --lpips-weight 0.1 --wd-sigma-mode static --wd-sigma-max 32.0 \
    --wd-sigma-p-min 0.5 --emlnet-blur --emlnet-norm --epochs 2 -lr 1e-4 --batch-size 8 \
    --patch-size 512 512 --msd-weight 0.02 \
    --save_path /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/ckpt_ssim/${name}.pth.tar \
    --save --skip-loss-threshold 5.0 \
    --comet --comet-name ${name}

