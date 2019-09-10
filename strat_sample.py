from __future__ import division
from __future__ import print_function

import os
os.environ['PYTHONHASHSEED'] = '2018'
glo_seed = 2018

import numpy as np
np.random.seed(glo_seed)
rng = np.random.RandomState(seed=glo_seed)

def iterative_sampling(Y, labeled_idx, fold, rng):
	'''
		Y -- label array (unlabeled  nodes have a zero vector)
		labeled_idx -- indicies for labeled nodes
		fold -- number of splits (partritions)
		rng -- random state
		
		Stractified iterative spliting of multi-labeled 
	'''
	ratio_per_fold = 1 / fold
	folds = [[] for i in range(fold)]
	number_of_examples_per_fold = np.array([(1 / fold) * np.shape(Y[labeled_idx, :])[0] for i in range(fold)])

	blacklist_samples = np.array([])
	number_of_examples_per_label = np.sum(Y[labeled_idx, :], 0)
	blacklist_labels = np.where(number_of_examples_per_label < fold)[0]
	desired_examples_per_label = number_of_examples_per_label * ratio_per_fold
	if blacklist_labels.shape[0] > 0:
		print("The following labels were removed because of having",
		 " less samples than number of partritions: ({}) some samples",
		 " might be removed to preseve data intergrety".format(blacklist_labels))

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
						labeled_idx[np.where(labeled_idx == index)] = -1
						total_index = total_index - index

			number_of_examples_per_label[min_label_index[0]] = max_label_occurance
		except:
			traceback.print_exc(file=sys.stdout)
			exit()

	Y = Y[:, sel_labels]

	return folds, Y, blacklist_samples
