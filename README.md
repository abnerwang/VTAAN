# VTAAN
by [Xiaoping Wang](http://blog.keeplearning.group/about/).  
## Introduction
Python (PyTorch) implementation of VTAAN tracker「 Wang, F., Wang, X., Tang, J. *et al.* VTAAN: Visual Tracking with Attentive Adversarial Network. *Cogn Comput* (2020). [https://doi.org/10.1007/s12559-020-09727-3 ](https://doi.org/10.1007/s12559-020-09727-3 ) 」.

This implementation is based on my another repository called [py-Vital, ](https://github.com/abnerwang/py-Vital) which is posted on the [project home](https://github.com/ybsong00/Vital_release) by the authors of VITAL tracker.  

If you want this code for personal use, please cite:  

    @InProceedings{nam2016mdnet,
    author = {Nam, Hyeonseob and Han, Bohyung},
    title = {Learning Multi-Domain Convolutional Neural Networks for Visual Tracking},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2016}
    }  
      
    @inproceedings{shi-nips18-DAT,
    author = {Pu, Shi and Song, Yibing and Ma, Chao and Zhang, Honggang and Yang, Ming-Hsuan},
    title = {Deep Attentive Tracking via Reciprocative Learning},
    booktitle = {Neural Information Processing Systems},
    year = {2018},
    }   
     
    @inproceedings{xiaopingwang-VTAAN,
    author = {Xiaoping Wang}, 
    title = {VTAAN: Visual Tracking with Attentive Adversarial Network}, 
    booktitle = {VTAAN tracker implemented by PyTorch}, 
    month = {August},
    year = {2019},
    }  

  

## Prerequisites
- python 3.6+
- opencv 3.0+
- [PyTorch 1.0+](http://pytorch.org/) and its dependencies

## Usage

### Tracking
```bash
 python tracking/run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```
 - You can provide a sequence configuration in two ways (see tracking/gen_config.py):
   - ```python tracking/run_tracker.py -s [seq name]```
   - ```python tracking/run_tracker.py -j [json path]```

### Pretraining
 - Download [VGG-M](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat) (matconvnet model) and save as "models/imagenet-vgg-m.mat"
 - Pretraining on VOT-OTB
   - Download [VOT](http://www.votchallenge.net/) datasets into "datasets/VOT/vot201x"
    ``` bash
     python pretrain/prepro_vot.py
     python pretrain/train_mdnet.py
    ```