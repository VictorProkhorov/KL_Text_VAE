# Experiments
Here we provide scripts to replicate the results we obtained in the paper. 

NOTE: Even though the results in Table 2 and Table 3 are not posible to replicate (due to the sampling procedure), we still provide the sripts we used to run the experiments. By running these scripts, even with slightly different numbers, one still shall observe a similar pattern presented in the tables.
## Table of contents

1. [Figure 1](#figure-1)
2. [Table 1](#table-1)
3. [Table 2](#table-2)
4. [Table 3](#table-3)
5. [Table 4](#table-4)


## Figure 1
Running this script should allow you to get the results we reported in Figure 1.

Note: we only provide the pretrained models for three values of C= 15, 60 and 100. It should be enough to confirm the tren we report.

Example: by running the script below one would obtain the results for the VAE (LSTM encoder and CNN decoder) model trained on the Yahoo corpus.

```
$ python3 experiment_figure_1.py --corpus Yahoo --model LSTM-CONV
```

Type for more available options:
```
$ python3 experiment_figure_1.py --help
```

## Table 1
Running this script should allow you to get the results we reported in Table 1.

Example: by running the script below one would obtain the results for the VAE (LSTM encoder and LSTM decoder) model trained on the CBT corpus.

```
$ python3 experiment_table_1.py --corpus CBT
```

Type for more available options:
```
$ python3 experiment_table_1.py --help
```

NOTE: accidentally, in the paper, we swap the rows for the reconstruction experiment between WIKI and WebText corpora.

## Table 2
Running this script should allow you to get the results we reported in Table 2.

Example: by running the script below one would obtain the results for the VAE (LSTM encoder and LSTM decoder) model trained on the CBT corpus.
```
$ python3 experiment_table_2.py --corpus CBT
```

Type for more available options:
```
$ python3 experiment_table_2.py --help
```

## Table 3
Running this script should allow you to get the results we reported in Table 3.

NOTE: This script is still to be developed. But, it allready contains functions that allow to generate artificial corpora.

Type for more available options:
```
$ python3 experiment_table_3.py --help
```

## Table 4
Running this script should allow you to get the results we reported in Table 4.

Example: by running the script below one would obtain the results for the VAE (LSTM encoder and LSTM decoder) model trained on the Wiki corpus with C = 3.
```
$ python3 experiment_table_4.py --C 3
```

Type for more available options:
```
$ python3 experiment_table_4.py --help
```
 

