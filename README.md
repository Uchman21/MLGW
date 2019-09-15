# Collaborative Graph Walk for Semi-supervised Multi-Label Node Classification

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

#### Input format
The MLGW model handles dataset with/without edge attributes but assumes no missing node attribute. Therefore, at minimums, the code requires that a `--dataset_dir` option is specified which specifies the following data files:

* <dataset_dir>/\<dataset>.edgelist -- An edgelist file describing the input graph. This is a two columned file with format `<source node>\t<target node>`.
* <dataset_dir>/\<dataset>_matrices/node_attr.npz -- A numpy-stored array of node features ordered according to the node index in the edgelist file.
* <dataset_dir>/\<dataset>_matrices/edge_attr.npz [optional] -- A numpy-stored array of edge features if available; ordered according to the edge appearance in the edgelist file.
* <dataset_dir>/\<dataset>_matrices/label.npz -- A numpy-stored binary array of node labels; ordered according to the node index in the edgelist file. A zero vector is used to represent unlabeled nodes. For instance [[0,1],[1,1],[0,0]] the last entry represents an unlabeled node.

To run the model on a new dataset, you need to make data files in the format described above.

#### Model variants
The user must also specify a --variant, the variants of which are described in detail in the paper:
* mlgw_i -- In this variant, we let the agents make independent walks on the graph without a global policy regularization. Thus, no information sharing among the agents.
* mlgw_r -- This variant uses the global policy for regularization in the cost function, but makes decisions on which node to move using the local policy output
* mlgw_r+ -- This variant uses the global policy for regularization inthe cost function and also for the decisions in the graph walk.

#### Using the outputs

To save the node embeddings and walk paths, please use the `--save_emb` and `--save_path` options respectively. The ouput will be stored in an "output" folder.


