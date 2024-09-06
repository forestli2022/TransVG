export CUDA_VISIBLE_DEVICES=0,1

# ReferItGame
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone ViT-B/32 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50


# # RefCOCO
python -m torch.distributed.launch --master_port=29501 --nproc_per_node=2 --use_env train.py --batch_size 8 --lr_text 0.00001 --aug_crop --aug_scale --aug_translate --backbone ViT-B/32 --dataset unc --max_query_len 20 --output_dir outputs/refcoco_r50_clip


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone ViT-B/32 --detr_model ./checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/refcoco_plus_r50 --epochs 180 --lr_drop 120


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone ViT-B/32 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50


# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone ViT-B/32 --detr_model ./checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r50
