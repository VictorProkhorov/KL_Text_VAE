# On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation

## Replicate Results

We provide pretrained models (all models were trained on a GPU) and data used in the paper. To replicate the results:

### Step 1:
Download the all the data and models:  https://drive.google.com/file/d/1CAlEWVsU3k_YApIq-Qqu0ojPx45lw5fh/view?usp=sharing

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
@InProceedings{prokhorov_etal:WNGT2019,
  author={Victor Prokhorov and Ehsan Shareghi and Yingzhen Li and  Mohammad T. Pilehvar and Nigel Collier},
  title={On the Importance of the Kullback-Leibler Divergence Term in Variational Autoencoders for Text Generation},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019},
  month={November},
  address={Hong Kong},
}  
```

## Licence

The code in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation. The code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.


## Contact info

For questions or more information please use the following:
* **Email:** vp361@cam.ac.uk 

