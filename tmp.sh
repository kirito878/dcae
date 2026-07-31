# python dump.py \
#     --checkpoint /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/ckpt_epoch/dcae_wd32_512_static_from0240_mse_20_tuning_msd02_multi_noise/iter_12000.pth.tar \
#     --image /home/at9529/ycw.cs14/dataset/TestImage/Kodak/20.png \
#     --cuda \
#     --save_path latent_dump

# python check_saliency.py \
#     --image /home/at9529/ycw.cs14/dataset/TestImage/Kodak/20.png\
#     --emlnet-imagenet-path /home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_imagenet.pth \
#     --emlnet-places-path   /home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_places.pth \
#     --emlnet-decoder-path  /home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_decoder.pth \
#     --sigma-max 32 --p-min 0.5 --cuda --norm --blur \
#     --save_path saliency_check

# python check_hf_mask.py --image /home/at9529/ycw.cs14/dataset/TestImage/Kodak/20.png --save_path hf_check

python check_maskA.py \
    --image /home/at9529/ycw.cs14/dataset/TestImage/Kodak/20.png \
    --emlnet-imagenet-path /home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_imagenet.pth \
    --emlnet-places-path   /home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_places.pth \
    --emlnet-decoder-path  /home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/res_decoder.pth \
    --cuda --save_path maskA_check