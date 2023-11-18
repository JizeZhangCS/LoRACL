args=(  main_simclr_vit.py --ckpt_folder simclr_vit

    # ddp related param
    --multiprocessing-distributed --dist-url 'tcp://localhost:10000' --world-size 1 --rank 0
    
    # fixed param
    -b 512 --momentum 0.9 --print-freq 10 --valid_precent 0.1 --enable_aug

    # epoch related
    --epochs 200  --lora_start_epoch 50 --lora_fix_epoch 200
    --lr_sche cos

    # arch related param
    -a vit_tiny --weights 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
    
    # optimization related param
    --lr 3e-5 --wd 1e-1 

    # lora related param
    --enable_lora 
    --lora_lr 3e-6 --lora_wd 1e-1 --zero_init 
    --weight_types 'qv' --rank_of_lora 16
    --lora_lr_adjust  
    # --coop 
    # --lora_type none --lora_scale 0.1 
    
    # moco related param
    --dim 128 --temp 0.2

    # dataset related param
    --data ../datasets/PACS/art_painting 
    --iid_val_data ../datasets/PACS/art_painting 
    --ood_val_data ../datasets/PACS/sketch,../datasets/PACS/cartoon,../datasets/PACS/photo 

    # resume related param
    # --start-epoch 0 --resume './checkpoints/'

    # mostly unuseful param
    -j 16 

    # deprecated param
    # --lora_always_q
)

CUDA_VISIBLE_DEVICES=2 python "${args[@]}"
CUDA_VISIBLE_DEVICES=2 python "${args[@]}"
CUDA_VISIBLE_DEVICES=2 python "${args[@]}"
