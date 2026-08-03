# 調整sigma_offsets, 讓LL偏fidelity, 高頻偏realism, 並且讓四個子帶均等起步
    # wd_fn = VGG16WaveletWassersteinDistortion(
    #     num_levels=5, dwt_levels=1,
    #     learnable_weights=False,          # 关掉,别让它自己推向 HH
    #     sigma_offsets=(-0.5, 0.0, 0.0, 0.0),  # HH 从 1.0 降到 0.5,别过度realism
    #     ll_weight_boost=0.3,              # LL 温和回补一点(不是之前的1.0)
    # ).to(DEVICE)
# name="dcae_wd32_512_static_from0240_mse_20_tuning_msd_gate_v5_only_nt"
# python train_2.py -d /home/at9529/ycw.cs14/Michael/MLIC/mlic_train_100k --cuda --checkpoint /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/0.0018checkpoint_best.pth.tar \
#     --lambda 1 --wd-weight 0.5 --mse-weight 0 --final-mse-weight 0.01 \
#     --skip-warmup --lpips-weight 0.00 --wd-sigma-mode static --wd-sigma-max 32.0 \
#     --wd-sigma-p-min 0.3 --emlnet-blur --emlnet-norm --epochs 1 -lr 1e-4 --batch-size 8 \
#     --patch-size 512 512 --msd-weight 0.5 \
#     --save_path /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/ckpt_ssim/${name}.pth.tar \
#     --save --skip-loss-threshold 5.0 \
#     --gate-weight 0.1 \
#     --gate-budget 0.13 \
#     --gate-tv-weight 0.00 \
#     --tv-weight 0.000 \
#     --comet --comet-name ${name}

name="dcae_wd32_512_static_from0240_mse_20_tuning_msd_gate_v5_all"
python train_all.py -d /home/at9529/ycw.cs14/Michael/MLIC/mlic_train_100k --cuda --checkpoint /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/ckpt_ssim/dcae_wd32_512_static_from0240_mse_20_tuning_msd_gate_v5_only_nt.pth.tar \
    --lambda 1 --wd-weight 0.21 --mse-weight 0 --final-mse-weight 5 \
    --skip-warmup --lpips-weight 0.1 --wd-sigma-mode static --wd-sigma-max 32.0 \
    --wd-sigma-p-min 0.3 --emlnet-blur --emlnet-norm --epochs 1 -lr 5e-5 --batch-size 6 \
    --patch-size 512 512 --msd-weight 0.5 \
    --save_path /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/ckpt_ssim/${name}.pth.tar \
    --save --skip-loss-threshold 5.0 \
    --gate-weight 0.1 \
    --gate-budget 0.13 \
    --gate-tv-weight 0.00 \
    --tv-weight 0.00 \
    --comet --comet-name ${name}
