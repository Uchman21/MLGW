from __future__ import division
from __future__ import print_function

import os
glo_seed = 2018
os.environ['PYTHONHASHSEED'] = '2018'
import argparse
import time
from scipy import sparse as sp
from numpy.random import seed
import random as rn
import itertools, tqdm
rn.seed(glo_seed)
seed(glo_seed)
import subprocess
import tensorflow as tf

# from RNN_walk_ml import RNN_walk
tf.set_random_seed(glo_seed)
import pdb
import gc
import numpy as np
glo_seed = 2018
np.random.seed(glo_seed)
rng = np.random.RandomState(seed=glo_seed)

def iterative_sampling(Y, labeled_idx, fold, rng):
	ratio_per_fold = 1 / fold
	# indecies = np.arange(np.shape(Y)[0])
	folds = [[] for i in range(fold)]
	number_of_examples_per_fold = np.array([(1 / fold) * np.shape(Y[labeled_idx, :])[0] for i in range(fold)])

	blacklist_samples = np.array([])
	number_of_examples_per_label = np.sum(Y[labeled_idx, :], 0)
	blacklist_labels = np.where(number_of_examples_per_label < fold)[0]
	print(blacklist_labels)
	desired_examples_per_label = number_of_examples_per_label * ratio_per_fold

	subset_label_desire = np.array([desired_examples_per_label for i in range(fold)])
	total_index = np.sum(labeled_idx)
	max_label_occurance = np.max(number_of_examples_per_label) + 1
	sel_labels = np.setdiff1d(range(Y.shape[1]), blacklist_labels)

	while total_index > 0:
		try:
			min_label_index = np.where(number_of_examples_per_label == np.min(number_of_examples_per_label))[0]
			for index in labeled_idx:
				if (Y[index, min_label_index[0]] == 1 and index != -1) and (min_label_index[0] not in blacklist_labels):
					m = np.where(
						subset_label_desire[:, min_label_index[0]] == subset_label_desire[:, min_label_index[0]].max())[
						0]
					if len(m) == 1:
						folds[m[0]].append(index)
						subset_label_desire[m[0], Y[index, :].astype(np.bool)] -= 1
						labeled_idx[np.where(labeled_idx == index)] = -1
						number_of_examples_per_fold[m[0]] -= 1
						total_index = total_index - index
					else:
						m2 = np.where(number_of_examples_per_fold[m] == np.max(number_of_examples_per_fold[m]))[0]
						if len(m2) > 1:
							m = m[rng.choice(m2, 1)[0]]
							folds[m].append(index)
							subset_label_desire[m, Y[index, :].astype(np.bool)] -= 1
							labeled_idx[np.where(labeled_idx == index)] = -1
							number_of_examples_per_fold[m] -= 1
							total_index = total_index - index
						else:
							m = m[m2[0]]
							folds[m].append(index)
							subset_label_desire[m, Y[index, :].astype(np.bool)] -= 1
							labeled_idx[np.where(labeled_idx == index)] = -1
							number_of_examples_per_fold[m] -= 1
							total_index = total_index - index
				elif (Y[index, min_label_index[0]] == 1 and index != -1):
					if (min_label_index[0] in blacklist_labels) and np.any(Y[index, sel_labels]) == False:
						np.append(blacklist_samples, index)

						# subset_label_desire[m,Y[index,:]] -= 1
						labeled_idx[np.where(labeled_idx == index)] = -1
						# number_of_examples_per_fold[m] -= 1
						total_index = total_index - index

			number_of_examples_per_label[min_label_index[0]] = max_label_occurance
		except:
			traceback.print_exc(file=sys.stdout)
			exit()

	Y = Y[:, sel_labels]

	return folds, Y, blacklist_samples



def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run Citelearn.")

	parser.add_argument('--dataset', type=str, default='cora',
						help='Dataset index [dblp, cora, cora_new, delve]')

	parser.add_argument('--model', type=str, default='ind',
						help='Model to use [ind, kl, mean]')

	parser.add_argument('--lr',  type=float, default=0.01,
						help='Learning rate [0.01, 0.001, 0.0001]')

	parser.add_argument('--kl_cost',  type=float, default=0.1,
						help='KL divergance cost penalty ')

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--num-walks', type=int, default=10,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('--transductive', dest='transductive', action='store_true',
						help='Boolean specifying if to train a transductive model. Default is False.')
	parser.set_defaults(transductive=False)

	parser.add_argument('--train_ratio', help='ratio of dataset to use as train set (50% default)', type=int,
						default=50)
	parser.add_argument('--epoch', help='number of training epoch (10 default)', type=int,
						default=10)
	parser.add_argument('--test_single', dest='test_single', action='store_true',
						help='Boolean specifying if to test only on each cv patrition(1/fold). Default is False.')
	parser.set_defaults(test_single=False)

	parser.add_argument('--save', dest='save', action='store_true',
						help='Boolean specifying if to save trained paths only. Default is False.')
	parser.set_defaults(save=False)

	return parser.parse_args()


def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''

	old_state = rn.getstate()
	walk_len = 40
	batchsize = 32
	nepochs = args.epoch


	global_best = -1
	gb_config = []
	_split = None

	_dataset = args.dataset
	lrate = args.lr
	kl_cost = args.kl_cost

	model_dir = "checkpoint/models"

	ratio = args.train_ratio / 100.0
	l_dim = 128
	max_neighbors = 40
	if ratio == 1:
		num_runs = 1
	else:
		num_runs = 5

	benchmark_ml(_dataset, l_dim, walk_len, max_neighbors, lrate, kl_cost, model_dir,ratio, num_runs,
			  is_toy=False, batchsize=batchsize, nepochs=nepochs, verbose=False, model=args.model, is_transductive=args.transductive)

	# for walk_len in tqdm.tqdm(walk_length):
	#
	# 	rn.setstate(old_state)
	#
	# 	l_dim_arr = [128]
	# 	L = 40
	# 	# if args.rf:
	# 	# 	lrate = 0.0002
	# 	# else:
	# 	# 	lrate = 0.0001
	#
	# 	if ratio == 1:
	# 		num_runs = 1
	# 	else:
	# 		if args.dataset in [2,0]:
	# 			num_runs = 2
	# 		else:
	# 			num_runs = 3
	#
	# 	for l_dim in l_dim_arr:
	# 		if args.ml == False:
	# 			best = benchmark(_dataset, l_dim, walk_len, L, lrate, model_dir,ratio, num_runs, is_toy=False, batchsize=batchsize, nepochs=nepochs, verbose=False)
	# 		else:
	# 			best = benchmark_ml(_dataset, l_dim, walk_len, L, lrate, model_dir,ratio, num_runs, is_toy=False,  batchsize=batchsize, nepochs=nepochs, verbose=False)
	#
	# 		if best > global_best:
	# 			global_best = best
	# 			gb_config.append([l_dim, walk_len])
	# 			# output_file.write("---------------------best so far --------------------\n")
	# 			# output_file.write("best accuracy: {}\n".format(global_best))
	# 			print("---------------------best so far --------------------\n")
	# 			print("best accuracy: {}\n".format(global_best))
	# 		elif best == global_best:
	# 			gb_config.append([l_dim, walk_len])



# output_file.write ("---------------------best general --------------------\n")
# output_file.write("best accuracy: {}\n".format(global_best))
# output_file.write("best configurations: h_dim:{}, walk_length: {}".format(
# 	gb_config[-1][0], gb_config[-1][1]))
# output_file.write("{}".format(gb2config))


def benchmark_ml(_dataset, l_dim, walk_len, L, lrate, kl_cost, model_dir, train_ratio, num_runs, is_toy=False, batchsize=128, nepochs = 100, verbose=False, model='ind', is_transductive=False):
	if model == 'ind':
		from RNN_walk_ml import RNN_walk
	elif model == 'kl':
		from RNN_walk_mlv2 import RNN_walk
	elif model =='mean':
		from RNN_walk_mlv4 import RNN_walk
	else:
		from RNN_walk_mlv4 import RNN_walk

	acc_nn, err = 0 , 0
	labels = sp.load_npz("../final_extraction/{}/{}_matrices/label.npz".format(_dataset,_dataset)).astype(np.int32)
	labels = sp.vstack((np.zeros(labels.shape[-1]), labels)).tocsr()
	dim_y = labels.shape[1]
	labeled_idx = np.where(labels.sum(-1) > 0)[0]
	out_file = open("outputs/{}_{}_{}_{}_{}.txt".format(_dataset, lrate, train_ratio, nepochs, is_transductive), 'w', 0)
	cv_splits, Y, blacklist_samples = iterative_sampling(labels.toarray(), labeled_idx, num_runs, rng)

	num_of_label = labels.shape[1]
	score_micro_final = np.zeros(3)
	score_macro_final = np.zeros(3)
	acc_final = 0

	precision_final = np.zeros(num_of_label)
	recall_final = np.zeros(num_of_label)
	F1_final = np.zeros(num_of_label)

	for i in range(len(cv_splits)):
		training_samples = []
		testing_samples = []
		for j in range(len(cv_splits)):
			if args.test_single:
				if j != i:
					training_samples += cv_splits[j]
				# elif j == ((i + 1) % folds):
				#     val_mask = cv_splits[j]
				# elif j == ((i + 2) % folds):
				#     testing_samples = cv_splits[j]
				else:
					testing_samples = cv_splits[j]
			else:
				if j == i:
					training_samples = cv_splits[j]
				else:
					testing_samples += cv_splits[j]
		# print("iter {} of {}..........".format(i+1, num_runs))
		gc.collect()
		tf.reset_default_graph()
		r_walk = RNN_walk(_dataset, l_dim, dim_y, walk_len, L, lrate, kl_cost, batchsize, model_dir, is_toy, verbose, is_transductive)
		if i == 0:
			print(r_walk)

		masks = [np.array(training_samples), np.array(testing_samples)] #train_mask, u_mask, test_mask
		r_walk.load_data(masks[-1])
		acc, score_micro, score_macro, per_label =  r_walk.train(labels, train_ratio, masks, nepochs, verbose, False)
		acc_final += acc
		precision_final += per_label[0]
		recall_final += per_label[1]
		F1_final += per_label[2]
		score_macro_final += score_macro
		score_micro_final += score_micro

		print("iter [{}] of [{}] : Acc {},  Macro_F1 {}".format(i+1, num_runs, acc, score_macro[-1]))
		out_file.write("iter [{}] of [{}] : Acc {},  Macro_F1 {}".format(i+1, num_runs, acc, score_macro[-1]))
		del r_walk
		gc.collect()
		exit()

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
	out_file.close()

# return acc_nn/num_runs
#
# def benchmark_ml(_dataset, l_dim, walk_len, L, lrate, model_dir, train_ratio, num_runs, is_toy=False, batchsize=128, nepochs = 100, verbose=False):
# 	acc_nn, acc_lg, err = 0 , 0, 0
# 	labels = sp.load_npz("../final_extraction/{}/{}_matrices/label.npz".format(_dataset,_dataset)).astype(np.int32)
# 	dim_y = labels.shape[1]
# 	labeled_idx = np.where(labels.sum(-1) > 0)[0]
# 	folds = 5
# 	cv_splits, Y, blacklist_samples = iterative_sampling(labels.toarray(), labeled_idx, folds, rng)
# 	# training_samples = []
# 	# testing_samples = []
# 	num_of_label = labels.shape[1]
# 	score_micro_final = np.zeros(3)
# 	score_macro_final = np.zeros(3)
# 	acc_final = 0
#
# 	precision_final = np.zeros(num_of_label)
# 	recall_final = np.zeros(num_of_label)
# 	F1_final = np.zeros(num_of_label)
#
# 	for i in range(len(cv_splits)):
# 		training_samples = []
# 		testing_samples = []
# 		for j in range(len(cv_splits)):
# 			if args.test_single:
# 				if j != i:
# 					training_samples += cv_splits[j]
# 				# elif j == ((i + 1) % folds):
# 				#     val_mask = cv_splits[j]
# 				# elif j == ((i + 2) % folds):
# 				#     testing_samples = cv_splits[j]
# 				else:
# 					testing_samples = cv_splits[j]
# 			else:
# 				if j == i:
# 					training_samples = cv_splits[j]
# 				else:
# 					testing_samples += cv_splits[j]
# 			# print("iter {} of {}..........".format(i+1, num_runs))
# 		tf.reset_default_graph()
# 		r_walk = RNN_walk(_dataset, l_dim, dim_y, walk_len, L, lrate, batchsize,  model_dir, is_toy, verbose)
# 		if i == 0:
# 			print(r_walk)
# 		r_walk.load_data()
# 		acc, score_micro, score_macro = r_walk.train(labels, train_ratio, [training_samples, testing_samples], nepochs, verbose,save=args.save)
# 		acc_final += acc
# 		# precision_final += precision
# 		# recall_final += recall
# 		# F1_final += F1
# 		score_macro_final += score_macro
# 		score_micro_final += score_micro
# 		print("iter [{}] of [{}] : Acc {},  Macro_F1 {}".format(i+1, folds, acc, score_macro[-1]))
#
# 	# precision_final /= folds
# 	# recall_final /= folds
# 	# F1_final /= folds
# 	score_micro_final /= folds
# 	score_macro_final /= folds
# 	acc_final /= folds
#
# 	# print("Precision : {0}\n".format(', '.join(
# 	# 	'l{0:1d}: {1:.4f}'.format(*k) for k in enumerate(precision_final.tolist()))))
# 	# print("Recall : {0}\n".format( ', '.join(
# 	# 	'l{0:1d}: {1:.4f}'.format(*k) for k in enumerate(recall_final.tolist()))))
# 	# print("F1 : {0}\n".format(', '.join('l{0:1d}: {1:.4f}'.format(*k) for k in enumerate(F1_final.tolist()))))
# 	print("Micro Score --> precision = {0}, recall = {1}, F1 == {2}\n".format(score_micro_final[0],
# 																					score_micro_final[1],
# 																					score_micro_final[2]))
# 	print("Macro Score --> precision = {0}, recall = {1}, F1 == {2}\n".format(score_macro_final[0],
# 																					score_macro_final[1],
# 																					score_macro_final[2]))
# 	print("Accuracy Score  --> subset accuracy = {0}\n".format( acc_final))
# 	print("Av_F1 = {0}\n".format(score_macro_final[2]))
#
# 	return acc_final
#

# learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
