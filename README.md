# FGSM for Autoeval

## PyTorch Implementation

This repository contains:

- the PyTorch implementation 
- the example on tinyimagenet setup
- Contrastive Accuracy calculation and linear regression methods
- FGSM based adversarial test examples generate

Please follow the instruction below to install it and run the experiment demo.

### Prerequisites
* Linux (tested on Ubuntu 16.04LTS)
* NVIDIA GPU + CUDA CuDNN (tested on Tesla V100)

* [TinyImageNet Dataset](http://cs231n.stanford.edu/tiny-imagenet-200.zip) (download and unzip to ```PROJECT_DIR/datasets/tiny-imagenet-200```)
* [TinyImageNet-C Dataset](https://zenodo.org/record/2469796#.Y-3gynZBx3g) (download and unzip to ```PROJECT_DIR/datasets/Tiny-ImageNet-C```)
* SHVN,USPS,FashionMNIST,KMNIST,STL10 are the ready-made datasets in torchvision.datasets package 
* All -C Operation can refers to [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://github.com/hendrycks/robustness)

## Getting started
0. Install dependencies 
    ```bash

    # CAME based autoeval
    conda env create --name contrastive_autoeval --file environment.yml

    ```

1. Co-train classifier
    ```bash
    # Save as "PROJECT_DIR/checkpoints/TinyImagenet/checkpoint.pth"
    python run.py
    ```
    
2. Creat Meta-set
    ```bash
    # By default it creates 200 sample sets
    python meta_set/synthesize_set_tinyimagenet.py
    ```
   
3. Test classifier on Meta-set and save the fitted regression model
    ```bash
    python regression.py
    ```
4. Prepare the adversarial test examples for autoeval
    ```bash
    # This file generate adversarial test examples by computing the gradient of true label in order to maximum its confidence.
    python prepare_advdata.py
    ``` 

5. Test on unseen test sets by regression model
    ```bash
    # 1) You will see the score of both the test set and its advsarial set.
    # 2) The absolute error of linear regression are also shown below.
    python test_attack.py
    ``` 

        
## Citation
If you use the code in your research, please cite:
```bibtex
    @inproceedings{peng2023contrastive,
    author={Jiachen, Ma},
    title     = {FGSM Attack For Autoeval},
    booktitle = {Proc. ICCV},
    year      = {2023},
    }
```

## License
MIT
