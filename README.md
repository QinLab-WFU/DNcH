# Deep Neighbor-coherence Hashing with Discriminative Sample Mining for Supervised Cross-modal Retrieval [Paper](https://www.sciencedirect.com/science/article/pii/S095741742500987X)
This paper is accepted for publication with Expert Systems With Applications.

## Training

### Processing dataset
Before training, you need to download the oringal data from [coco](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)(include 2017 train,val and annotations), [nuswide](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)(include all), mirflickr25k [Baidu, 提取码:u9e1](https://pan.baidu.com/s/1upgnBNNVfBzMiIET9zPfZQ) or [Google drive](https://drive.google.com/file/d/18oGgziSwhRzKlAjbqNZfj-HuYzbxWYTh/view?usp=sharing) (include mirflickr25k and mirflickr25k_annotations_v080), 
then use the "data/make_XXX.py" to generate .mat file

After all mat file generated, the dir of `dataset` will like this:
~~~
dataset
├── base.py
├── __init__.py
├── dataloader.py
├── coco
│   ├── caption.mat 
│   ├── index.mat
│   └── label.mat 
├── flickr25k
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
└── nuswide
    ├── caption.txt  # Notice! It is a txt file!
    ├── index.mat 
    └── label.mat
~~~

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

## Training

### Processing dataset
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.0001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 64

## Citation
@article{ZHU2025127365,
title = {Deep neighbor-coherence hashing with discriminative sample mining for supervised cross-modal retrieval},
journal = {Expert Systems with Applications},
pages = {127365},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.127365},
url = {https://www.sciencedirect.com/science/article/pii/S095741742500987X},
author = {Congcong Zhu and Qibing Qin and Wenfeng Zhang and Lei Huang},
keywords = {Deep hashing, Supervised cross-modal retrieval, Neighbor-aware constraint, Multi-modal hard pair, Dynamic weight},
abstract = {Deep supervised cross-modal hashing has attracted extensive attention because of its low cost and high retrieval efficiency. Although the existing deep supervised cross-modal hashing methods have made great progress, they still suffer from two factors in the preservation of semantic relations between heterogeneous modalities. (1) Most of the available deep supervised cross-modal hashing learn hash functions by employing either pair-wise/multi-wise loss to explore the point-to-point relation or class center loss to explore the point-to-class relation, ignoring collaborative semantic relations. (2) Compared with the large proportion of simple samples, the hard pairs with a small proportion could provide more valuable information for the model training, nevertheless, most deep hash treats all samples equally in the learning process, and overlooks the positive contribution of hard samples in the learning process, impeding the hash function learning. To address these challenges, by considering both point-to-point and point-to-class relations, the novel Deep Neighbor-coherence Hashing (DNcH) framework is proposed to preserve the consistency of neighbor relations and generate high-quality binary codes with intra-class compactness and inter-class separability. Specifically, by jointly exploring the point-to-point and point-to-class relations between heterogeneous data, the neighbor-aware constraint is proposed to project the heterogeneous data into a unified Hamming space, where each anchor is close to all similar samples and corresponding class center, and far away from dissimilar samples and their class centers. The hard pairs containing valuable information are effectively mined by introducing the multi-similarity measurement strategy between heterogeneous modalities to construct the informative and representative training batches. Besides, to further gradually capture discriminant information from multi-modal hard pairs, a self-paced learning mechanism is introduced to assign dynamic weights to multi-modal pairs, which enables the deep cross-modal hashing to gradually concentrate on hard pairs while jointly learning universal patterns from the entire set of multi-modal pairs. Extensive experiments on three benchmark datasets show that our DNcH framework has better performance than the most advanced cross-modal hashing methods. The source code for the DNcH framework is available at https://github.com/QinLab-WFU/DNcH.}
}

