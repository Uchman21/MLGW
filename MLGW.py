from __future__ import division
from __future__ import print_function

import os
os.environ['PYTHONHASHSEED'] = '2018'
glo_seed = 2018

import argparse
import time
from scipy import sparse as sp
import random as rn
import itertools, tqdm
import subprocess
import tensorflow as tf
import numpy as np
import gc
from strat_sample import iterative_sampling

rn.seed(glo_seed)
np.random.seed(glo_seed)
tf.set_random_seed(glo_seed)

rng = np.random.RandomState(seed=glo_seed)


def parse_args():
	'''
	Parses the MLGW arguments (setup).
	'''

	# Settings
	flags = tf.app.flags
	FLAGS = flags.FLAGS

	flags.DEFINE_string('dataset','dblp_multi','Dataset index [dblp, cora, cora_new, delve].')
	flags.DEFINE_string('variant', 'mlgw_i', 'variant to use [mlgw_i, mlgw_kl, mlgw_mean].')
	flags.DEFINE_float('lrate',  0.01, 'Learning rate.')
	flags.DEFINE_float('gamma',  0.9, 'Reinforcement learning reward discount rate')
	flags.DEFINE_float('beta',1, 'KL divergance strenght. Used only for mlgw_kl variant.')
	flags.DEFINE_float('alpha',0.1, 'Entropy strenght. Used only for mlgw_kl variant.')
 	flags.DEFINE_integer('l_dim', 128, 'Dimension size of the latent vectors. Default is 128.')
	flags.DEFINE_integer('batchsize', 32,'Size of batch input. Default is 128.')
	flags.DEFINE_integer('max_neighbors', 40, 'maximum node neighbors to consider per step.')
	flags.DEFINE_integer('num_walks', 3,'Number of walks per source. Default is 3.')
	flags.DEFINE_integer('walk_len',20,'Number of nodes to tranverse per walk. Default is 20.')
	flags.DEFINE_boolean('transductive', False, 'Boolean specifying if to train a transductive model. Default is False.')
	flags.DEFINE_float('train_ratio', 0.5, 'ratio of dataset to use as train set (0.5 default).')
	flags.DEFINE_integer('epochs', 10, 'number of training epoch (10 default).')
	flags.DEFINE_boolean('test_single', False, 'Boolean specifying to test on each cv patrition(1/fold). Default is False.')
	flags.DEFINE_boolean('save',False, 'Boolean specifying if to save trained paths only. Default is False.')
	flags.DEFINE_boolean('verbose',False, 'display all outputs. Default is False.')
	flags.DEFINE_boolean('has_edge_attr',True, 'The dataset has edge attributes. Default is True.')
	

	return FLAGS


def main():
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''

	FLAGS = parse_args()

	benchmark_ml(FLAGS)


def benchmark_ml(FLAGS):

	from model import RNN_walk
	
	#initializations
	
	acc_final = 0
	acc_nn, err = 0 , 0
	patritions = 5
	num_runs = 0
	labels = sp.load_npz("../final_extraction/{}/{}_matrices/label.npz".format(FLAGS.dataset,FLAGS.dataset)).astype(np.int32)
	labels = sp.vstack((np.zeros(labels.shape[-1]), labels)).tocsr()
	labeled_idx = np.where(labels.sum(-1) > 0)[0]
	score_micro_final = np.zeros(3)
	score_macro_final = np.zeros(3)

	

	#split the labeled datasets into partritions using iterative stratified sampling
	cv_splits, labels, blacklist_samples = iterative_sampling(labels, labeled_idx, patritions, rng)
	num_of_labels = labels.shape[1]
	precision_final = np.zeros(num_of_labels)
	recall_final = np.zeros(num_of_labels)
	F1_final = np.zeros(num_of_labels)
	
	#training and evaluations

	for i in range(1):  #change range(1) to range(len(cv_splits)) to evaluate on all splits

		#setup the training and testing set per iteration
		training_samples = []
		testing_samples = []
		for j in range(len(cv_splits)):
			if FLAGS.test_single:
				if j != i:
					training_samples += cv_splits[j]
				else:
					testing_samples = cv_splits[j]
			else:
				if j == i:
					training_samples = cv_splits[j]
				else:
					testing_samples += cv_splits[j]

		gc.collect()  # force gabage collect to clean up memory
		tf.reset_default_graph()  #rest the Tensowflow graph
		
		r_walk = RNN_walk(FLAGS, num_of_labels)
		if i == 0:
			print(r_walk)

		samples = [np.array(training_samples), np.array(testing_samples)] #train_samples, test_samples
		
		#load and setup the node and edge attributes
		r_walk.load_data(samples[-1])

		#train and evaluate
		acc, score_micro, score_macro, per_label =  r_walk.train(labels, samples)
		
		#gather evaluation scores
		acc_final += acc
		precision_final += per_label[0]
		recall_final += per_label[1]
		F1_final += per_label[2]
		score_macro_final += score_macro
		score_micro_final += score_micro

		if FLAGS.verbose:
			print("iter [{}] of [{}] : Acc {},  Macro_F1 {}".format(i+1, num_runs, acc, score_macro[-1]))
		del r_walk
		num_runs += 1

	score_micro_final /= num_runs
	score_macro_final /= num_runs
	acc_final /= num_runs

	precision_final /= num_runs
	recall_final /= num_runs
	F1_final /= num_runs

	print("Micro Score --> precision = {0}, recall = {1}, F1 == {2}\n".format(score_micro_final[0],
																			  score_micro_final[1],
																			  score_micro_final[2]))
	print("Macro Score --> precision = {0}, recall = {1}, F1 == {2}\n".format(score_macro_final[0],
																			  score_macro_final[1],
																			  score_macro_final[2]))
	print("Accuracy Score  --> subset accuracy = {0}\n".format( acc_final))
	print("Av_F1 = {0}\n".format(score_macro_final[2]))
	print("\n----------------per Label pecision -----------------")
	print(precision_final)
	print("\n----------------per Label recall -----------------")
	print(recall_final)
	print("\n----------------per Label f1 -----------------")
	print(F1_final)


if __name__ == "__main__":
	main()