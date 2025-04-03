#  [Deep Neighbor-coherence Hashing with Discriminative Sample Mining for Supervised Cross-modal Retrieval](https://www.sciencedirect.com/science/article/pii/S095741742500987X)
This paper is accepted for publication with Expert Systems With Applications.

## Training

### Processing dataset
Before training, you need to download the oringal data from [coco](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)(include 2017 train,val and annotations), [nuswide](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)(include all),  [IAPRTC12](https://www.kaggle.com/datasets/parhamsalar/iaprtc12) , 
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
├── iapr
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


### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.0001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 64

## Citation
@article{zhu2025deep,  
  title={Deep neighbor-coherence hashing with discriminative sample mining for supervised cross-modal retrieval},  
  author={Zhu, Congcong and Qin, Qibing and Zhang, Wenfeng and Huang, Lei},  
  journal={Expert Systems with Applications},  
  pages={127365},  
  year={2025},  
  publisher={Elsevier}  
}
