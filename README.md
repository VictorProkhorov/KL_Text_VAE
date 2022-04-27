# [On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation](https://arxiv.org/abs/1909.13668)

## Table of contents

1. [Replicate Results](#replicated-results)
2. [Usage](#usage)
3. [Citing](#citing)
4. [Licence](#licence)
5. [Contact info](#contact-info)


## Replicate Results

We provide pretrained models (all models were trained on a GPU) and data used in the paper. To replicate the results:

### Step 1:
1. Download all the data and models:  https://drive.google.com/file/d/1CAlEWVsU3k_YApIq-Qqu0ojPx45lw5fh/view?usp=sharing

2. Once downloaded move the folder to the ./KL_Text_VAE directory 

### Step 2:
```
$ cd ./Scripts/Experiments/
```
all the instructions are provided in the directory.

## Usage

To train a new model:
```
$ cd ./Scripts/Model/
$ python3 model_wngt.py --corpus <name_of_corpus> --C <value of C> --checkpoint <save_model_file> --is_load 0 --model <model to train: e.g. LSTM-LSTM>
```
For more info execute:
```
$ python3 model_wngt.py --help
```

## Citing

If you find this material useful in your research, please cite:

```
@inproceedings{prokhorov-etal-2019-importance,
    title = "On the Importance of the {K}ullback-{L}eibler Divergence Term in Variational Autoencoders for Text Generation",
    author = "Prokhorov, Victor  and
      Shareghi, Ehsan  and
      Li, Yingzhen  and
      Pilehvar, Mohammad Taher  and
      Collier, Nigel",
    booktitle = "Proceedings of the 3rd Workshop on Neural Generation and Translation",
    month = nov,
    year = "2019",
    address = "Hong Kong",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-5612",
    doi = "10.18653/v1/D19-5612",
    pages = "118--127",
    abstract = "Variational Autoencoders (VAEs) are known to suffer from learning uninformative latent representation of the input due to issues such as approximated posterior collapse, or entanglement of the latent space. We impose an explicit constraint on the Kullback-Leibler (KL) divergence term inside the VAE objective function. While the explicit constraint naturally avoids posterior collapse, we use it to further understand the significance of the KL term in controlling the information transmitted through the VAE channel. Within this framework, we explore different properties of the estimated posterior distribution, and highlight the trade-off between the amount of information encoded in a latent code during training, and the generative capacity of the model.",
}
```

## Licence

The code in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation. The code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.


## Contact info

For questions or more information please use the following:
* **Email:** victorprokhorov91@gmail.com 

