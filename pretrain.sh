torchrun --standalone --nproc_per_node=8 --master_port 1221 main_pretrain.py \
--batch_size 128 --accum_iter 2 \
    --model arm_base_pz16 \
    --norm_pix_loss \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /data1/data/ImageNet/ --output_dir ./out_b/
