args=(  main_simclr.py --ckpt_folder simclr

    # ddp related param
    --multiprocessing-distributed --dist-url 'tcp://localhost:29995' --world-size 1 --rank 0    
    # fixed param
    -b 512 --momentum 0.9 --print-freq 10 --valid_precent 0.1 --enable_aug

    # epoch related
    --epochs 200  --lora_start_epoch 50 --lora_fix_epoch 200
    --lr_sche cos

    # arch related param
    -a resnet18 --weights 'DEFAULT'
    
    # optimization related param
    --lr 3e-3 --wd 1e-4 

    # lora related param
    --enable_lora 
    --lora_lr 3e-3 --lora_wd 1e-4 --zero_init 
    --lora_layers 12 --rank_of_lora 16
    --lora_lr_adjust  
    # --coop 
    # --lora_type none --lora_scale 0.1 
    
    # simclr related param
    --dim 128 --temp 0.2

    # dataset related param
    --data ../datasets/PACS/art_painting 
    --iid_val_data ../datasets/PACS/art_painting 
    --ood_val_data ../datasets/PACS/sketch,../datasets/PACS/cartoon,../datasets/PACS/photo 

    # resume related param
    # --start-epoch 0 --resume './checkpoints/'

    # mostly unuseful param
    -j 16 

)

source your/path/to/conda.sh
conda activate loracl
python "${args[@]}"
python "${args[@]}"
python "${args[@]}"
