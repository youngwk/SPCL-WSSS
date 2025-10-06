# Leveraging Single Positive Class Label Supervision for Weakly Supervised Semantic Segmentation

## Abstract

Weakly supervised semantic segmentation (WSSS) aims to reduce the annotation costs of training semantic segmentation networks by leveraging weak supervision. Although class labels are a widely used form of weak supervision, annotating all object categories within scene-level images remains both labor-intensive and error-prone, since annotators must exhaustively verify the presence of multiple classes. To overcome this, we propose a novel WSSS framework that relies on only a single positive class label per image as supervision, substantially simplifying the annotation process and enabling more scalable dataset construction. However, using single positive labels introduces challenges, as they can lead to degraded pseudo-semantic masks and hinder network training. To address this, we propose Prediction Score Thresholding (PST) and Permanently-corrected Label Transfer (PLT) to alleviate the degradation problem. Experimental results on PASCAL VOC 2012 and Microsoft COCO 2014 demonstrate that our proposed methods significantly enhance both the quality of the pseudo-semantic mask and the overall WSSS performance, even with extremely sparse supervision.


## Prerequisite
- Python 3.6, PyTorch 1.9, and others in requirements.txt
- We recommend using docker to create environment via Dockerfile

## Usage

### (Note) This branch contains the code for reproducing experiments based on AMN. For experiments based on IRN, please refer to the "irn" branch.

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
python run_sample_pascal.py --work_space YOUR_WORK_SPACE --voc12_root YOUR_DATASET_DIRECTORY --loss_type llcp --activation_type pstplt 
```

#### MS COCO
```
python run_sample_coco.py --work_space YOUR_WORK_SPACE --coco14_root YOUR_DATASET_DIRECTORY --loss_type llcp --activation_type pstplt
```

### Step 4. Train semantic segmentation network.

To train DeepLab-v2, we refer to [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). 
We use the [ImageNet pre-trained model](https://drive.google.com/file/d/14soMKDnIZ_crXQTlol9sNHVPozcQQpMn/view?usp=sharing) for DeepLabV2 provided by [AdvCAM](https://github.com/jbeomlee93/AdvCAM).
Please replace the ground truth masks with generated pseudo masks.

## Acknowledgment
Our code is heavily built upon [ReCAM](https://github.com/zhaozhengChen/ReCAM) and [AMN](https://github.com/gaviotas/AMN).