# Single Positive Class Label in Weakly Supervised Semantic Segmentation


## Prerequisite
- Python 3.6, PyTorch 1.9, and others in environment.yml
- You can create the environment from environment.yml file
```
conda env create -f environment.yml
```

## Usage

### Step 1. Prepare dataset.

#### PASCAL VOC
- Download PASCAL VOC 2012 devkit from [official website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit). [Download](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). 
- You need to specify the path ('voc12_root') of your downloaded devkit in the following steps.

#### MS COCO
- Download MS COCO images from the [official COCO website](https://cocodataset.org/#download).
- You need to specify the path ('coco14_root') of your downloaded data in the following steps.
- Generate mask from annotations (annToMask.py file in ./coco14/).

### Step 2. Generate single positive class labels.

#### PASCAL VOC
```
cd voc12
python make_single_positive_cls_labels.py
```

#### MS COCO
```
cd coco14
python make_single_positive_cls_labels.py
```

### Step 3. Train both multi-label classification network and refinement network, and then generate pseudo-semantic masks.

#### PASCAL VOC
```
python run_sample_pascal.py --work_space YOUR_WORK_SPACE --loss_type llcp --delta_rel 0.002 --activation_type pstplt --pred_th 0.5
```

#### MS COCO
```
python run_sample_coco.py --work_space YOUR_WORK_SPACE --loss_type llcp --delta_rel 0.002 --activation_type pstplt --pred_th 0.5
```

### Step 4. Train semantic segmentation network.

To train DeepLab-v2, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). 
We use the [ImageNet pre-trained model](https://drive.google.com/file/d/14soMKDnIZ_crXQTlol9sNHVPozcQQpMn/view?usp=sharing) for DeepLabV2 provided by [AdvCAM](https://github.com/jbeomlee93/AdvCAM).
Please replace the ground truth masks with generated pseudo masks.

## Acknowledgment
Our code is heavily built upon [ReCAM](https://github.com/zhaozhengChen/ReCAM) and [AMN](https://github.com/gaviotas/AMN).