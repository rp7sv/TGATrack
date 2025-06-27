# Template-Guided Low-Rank Adaption for Robust RGB-T Tracking [ICME'2025 Oral]
Official implementation of [**TGATtrack**](https://arxiv.org/abs/2303.10826), including models and training&testing codes.

[Models & Raw Results](https://drive.google.com/drive/folders/19RjO_cabtJLzbIoYqDM4JPFjARrkidMD?usp=drive_link)
(Google Driver)
[Models & Raw Results](https://pan.baidu.com/s/1ec6Zz9zdb8uD0lqY3MnOYQ), 提取码：SSIC
(Baidu Driver)

:fire::fire::fire: This work proposes TGATrack, a new PEFT framework for RGB-T tracking.

## News
**[2025.06.27]**
- We release codes, model, training log and raw result.

## Introduction
- :fire: A new rboust PEFT RGB-T tracking framework.

- TGATrack has strong performance on RGB-T tracking task.

- TGATrack is with high parameter-efficient tuning, containing only 0.64M trainable parameters (<1%).

- We expect TGATrack can provide insights :fire: for further research of multi-modal tracking.


## Usage
### Installation
Create and activate a conda environment:
```
conda create -n tgatrack python=3.8
conda activate tgatrack
```
Install the required packages:
```
conda env create -f environment.yml
```

### Data Preparation
- Download [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR) and [RGBT234](https://github.com/mmic-lcl/Datasets-and-benchmark-code)

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_tgatrack>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Dowmload the pretrained [foundation model](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing) (OSTrack) 
and put it under ./pretrained/.
```
bash train.sh
```

### Testing
[LasHeR & RGBT234] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_rgbt.sh
```
We refer you to [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, 
and refer you to [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.

## Bixtex
If you find TGATrack is helpful for your research, please consider citing:

```bibtex

```

## Acknowledgment
- This repo is based on [OSTrack](https://github.com/botaoye/OSTrack), which is an excellent work.
- We thank for the [ViPT](https://github.com/jiawen-zhu/ViPT), which helps us to quickly implement our ideas.

## Contact
If you have any question, feel free to email binbing2024@outlook.com. ^_^