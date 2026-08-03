
# ckpt="2.4checkpoint_best"
# dataset="/home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/Kodak"
# CUDA_VISIBLE_DEVICES='0' python eval.py --checkpoint /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/ckpt_ssim/${ckpt}.pth.tar \
#      --data ${dataset} --cuda \
#      --save_path /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/results/ssim/${ckpt} \

# python /home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/eval_quality.py -r ${dataset} \
#     -i /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/results/ssim/${ckpt} \
#     -o /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/results/ssim/${ckpt}/quality_results.csv


ckpt="dcae_wd32_512_static_from0240_mse_20_tuning_msd_gate_v5_all_margin01"
epoch_num="iter_300.pth"
dataset="/home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/Kodak"
# dataset="/home/at9529/ycw.cs14/Michael/massive_activation/DCAE/img_20"
CUDA_VISIBLE_DEVICES='0' python eval.py --checkpoint /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/ckpt_epoch/${ckpt}/${epoch_num}.tar \
     --data ${dataset} --cuda \
     --save_path /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/results/ssim/${ckpt} \

python /home/at9529/ycw.cs14/Michael/massive_activation/ICLR2024-FTIC/eval_quality.py -r ${dataset} \
    -i /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/results/ssim/${ckpt} \
    -o /home/at9529/ycw.cs14/Michael/massive_activation/DCAE/results/ssim/${ckpt}/quality_results.csv
