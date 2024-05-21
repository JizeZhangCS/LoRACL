# LoRACL - Low-Rank Adversarial Contrastive Learning for Unlabeled Domain Generalization

## how to start

1. installing environment from environment.yml
2. prepare datasets
3. `bash ./script/simclr_vit_art.sh`

## datasets preparation

### file structure
```
base_folder/  
|–– LoRACL/  
|–– datasets/  
|   |–– /PACS
|   |   |–– /art_painting  
|   |   |–– /cartoon
|   |   |–– /photo
|   |   |–– /sketch  
|   |–– /DomainNet
|   |   |–– /clipart
|   |   |–– /clipart_test
|   |   |–– /infograph  
|   |   |–– /infograph_test
|   |   |–– ...
```

### link to datasets
[PACS](https://www.kaggle.com/datasets/nickfratto/pacs-dataset)
[DomainNet](http://ai.bu.edu/M3SDA/)

## Things to notice

DO rememeber preprocess DomainNet before using!
- train-test split
- remove unused categories according to Table 18 in [DARLING](https://arxiv.org/pdf/2107.06219)
