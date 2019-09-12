# Collaborative-Graph-Walk-for-Semi-supervised-Multi-Label-Node-Classification

### Overview

This directory contains code necessary to run the MLGW algorithm.
MLGW is an algorithm for semi-supervised multi-label node classification under the framework of reinforcement learning, which aims at integrating information from both labeled and unlabeled nodes into the learning process of node embeddings in attributed graphs. The learned node
embeddings are used to conduct both transductive and induc-tive multi-label node classification.

See our [paper](link will be provided on publication) for details on the algorithm.

The dblp directory contains the preprocessed dblp data used in our experiments.
The raw dblp and delve datasets (used in the paper) will be provided soon ().

If you make use of this code or the MLGW algorithm in your work, please cite the following paper:

	@inproceedings{akujumulti,
	     author = {Akujuobi, Uchenna and Yufei, Han and Zhang, Qiannan and Zhang, Xiangliang},
	     title = {Collaborative Graph Walk for Semi-supervised Multi-Label Node Classification},
	     booktitle = {ICDM},
	     year = {2019}
	  }

### Requirements

Recent versions of TensorFlow, numpy, scipy, sklearn are required. You can install all the required packages using the following command:

	$ pip install -r requirements.txt


### Running the code

Use `python MLGW.py` to run using default settings. The parameters can be changed by passing during the command call (e.g., `python MLGW.py --variant mlgw_i --verbose`). Use `python MLGW.py --help` to display the parameters.
