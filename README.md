# Curbing Task Interference using Representation Similarity-Guided Multi-Task Feature Sharing

OUTPUT_DIR: Directory to save output contents. <br />
DATA_DIR: Directory containing the datasets. <br />
MODEL_DIR: Directory containing the trained models. <br />

## Environment:

conda_env_local.yml file can be used to create an anaconda environment to run the code.

## Training script:

To train the One-De model on cityscapes dataset:

    python train.py --batch-size 8 --workers 8 --data-folder /DATA_DIR/Cityscapes --crop-size 512 1024 --checkname train_cs --config-file ./model_cfgs/cityscapes/one_de.yaml --epochs 140 --lr .0001 --output-dir OUTPUT_DIR --lr-strategy stepwise --lr-decay 98 126 --base-optimizer RAdam --dataset cityscapes
<br />
Other model configs can be found in 'model_cfgs' directory.


## Eval models:

Models can be evaluated using --eval-only arg along with train script.


## Get CKA similarities and task groupings:  
The following code runs grouping using seperate decoder (Sep-De). <br />

    python explain.py --batch-size 4 --workers 0 --crop-size 480 640 --config-file ./model_cfgs/cityscapes/sep_de_group.yaml --resume MODEL_DIR/model_latest_140.pth --data-folder /DATA_DIR/NYUv2 --data-folder-1 /DATA_DIR/NYUv2/image/train --explainer-name CKA --compare-tasks --dataset cityscapes

## Cite Our Work

If you find the code concerning Progressive Decoder Fusion (PDF) useful in your research, please consider citing our paper: 

    @InProceedings{pmlr-v199-gurulingan22a,
      title = 	 {Curbing Task Interference using Representation Similarity-Guided Multi-Task Feature Sharing},
      author =       {Gurulingan, Naresh Kumar and Arani, Elahe and Zonooz, Bahram},
      booktitle = 	 {Proceedings of The 1st Conference on Lifelong Learning Agents},
      pages = 	 {937--951},
      year = 	 {2022},
      editor = 	 {Chandar, Sarath and Pascanu, Razvan and Precup, Doina},
      volume = 	 {199},
      series = 	 {Proceedings of Machine Learning Research},
      month = 	 {22--24 Aug},
      publisher =    {PMLR},
      pdf = 	 {https://proceedings.mlr.press/v199/gurulingan22a/gurulingan22a.pdf},
      url = 	 {https://proceedings.mlr.press/v199/gurulingan22a.html},
    }

