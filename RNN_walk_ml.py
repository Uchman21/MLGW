
# -*- coding: utf-8 -*-

'''
LICENSE: BSD 2-Clause

Summer 2017
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, time
os.environ['PYTHONHASHSEED'] = '2018'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

glo_seed = 2018

import random as rn

rn.seed(glo_seed)
import pdb

import tensorflow as tf
import numpy as np
import itertools
import math
from scipy.sparse import hstack, csr_matrix, lil_matrix, vstack
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import load_npz, save_npz, csr_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import itertools, tqdm
np.random.seed(glo_seed)
tf.set_random_seed(glo_seed)
import shutil
# import RAW as rw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances



class RNN_walk():
    '''
    RNN attention walk
    '''

    def __init__(self, _dataset,l_dim, dim_y, walk_len, L, lrate, kl_cost, batch_size, model_dir, is_toy, verbose, is_transductive):
        '''
        Initializer
        _dataset -- name of the dataset to use
        dim_Av -- dimension of node features
        dim_Ae -- dimension of edge features
        dim_y -- number of classes
        l_dim -- hidden layer dimension
        walk_len -- length of walk
        L -- #edge samples to consider per node (e.g. 10)
        lrate -- learning rate (e.g. 1e-3) for Adam
        model_dir -- directory to save the trained models
        '''

        self.dtype = tf.float32
        self.model_dir = model_dir
        self._dataset = _dataset
        self.is_toy = is_toy
        self.verbose = verbose

        # the hyper-parameters
        self.config = OrderedDict()

        self.config['l_dim'] = l_dim
        self.config['walk_len'] = walk_len

        rn.seed(glo_seed)
        self.rng = np.random.RandomState(seed=glo_seed)
        np.random.seed(glo_seed)
        tf.set_random_seed(glo_seed)
        self.config['kl_cost'] = kl_cost

        self.config['L'] = L
        self.config['lrate'] = lrate
        self.config['trainA'] = 3
        self.config['testA'] = 3
        self.config['dim_y'] = dim_y
        self.config['is_transductive'] = is_transductive

        self.batch_size = batch_size


        self.config['activation'] = tf.nn.relu

        # place holders
        self.X = tf.placeholder(tf.int64, shape=(None,))
        # self.Xu = tf.placeholder(tf.int32, shape=(None,))

        self.is_training = tf.placeholder(tf.bool)
        self.get_path = tf.placeholder(tf.bool)

        rn.seed(glo_seed)
        self.rng = np.random.RandomState(seed=glo_seed)
        np.random.seed(glo_seed)
        tf.set_random_seed(glo_seed)

        self.config['activation'] = tf.nn.relu

        # place holders
        # for early stopping
        self.config['gamma'] = tf.constant(0.95)
        self.T = tf.constant(float(walk_len))
        self.range = tf.Variable(tf.range(0, self.config['walk_len'], 1, dtype=self.dtype), trainable=False)

        # self.GRU_cell_node()
        # self.GRU_cell_edge()
        # self.GRU_cell_bw()
        # self.GRU_cell_bw_edge()





        # identifier
        self.id = ('{0}_{1}_{2}_{3}_{4}'.format(
            self._dataset,
            self.config['walk_len'],
            self.config['l_dim'],
            self.config['L'],
            time.time()))

        # for early stopping
        self.min_loss = 100
        self.patience = 10
        self.steps_no_increase = 0
        self.is_train = True
        self._type = "peep"
        self.save_path = '{}/{}'.format(self.model_dir, self.id)


    def get_citecount(self, edge_conext_file):

        ref_c = []
        max = 0

        fp = open(edge_conext_file, "rU").read().split("\n")[:-1]
        for line in fp:
            edge = line.split("\t")
            if len(edge) > 3:
                context_count = len(edge[2:])
            elif len(edge) < 3:
                context_count = 1
            else:
                if edge[2] is not "":
                    context_count = len(edge[2].split(". "))
                else:
                    context_count = 1
            if max < context_count:
                max = context_count
            #
            # if context_count > 4:
            #     context_count = 3
            # elif context_count > 1:
            #     context_count = 2
            # elif context_count > 1:
            #     context_count = 1

            ref_c.append(context_count)
        ref_c = np.vstack((np.zeros(1), np.expand_dims(np.array((ref_c)),-1)/max))

        return ref_c

    def train_edge_weigths_cosine(self):
        features = load_npz("../final_extraction/{}/{}_matrices/lsi_ngram.npz".format(self._dataset,self._dataset)).toarray()
        features = Normalizer().fit_transform(features)
        input = open("../final_extraction/{}/{}.edgelist".format(self._dataset,self._dataset), "rU").read().split("\n")[:-1]
        fpw = open("../final_extraction/{}/{}_weighted.edgelist".format(self._dataset,self._dataset), "w")

        for i in range(len(input)):
            s, t = input[i].split("\t")
            weigth = 2 - cosine_distances([features[int(s)]], [features[int(t)]]).flatten()[0]
            fpw.write("{0:s}\t{1:.3f}\n".format(input[i], weigth))

    def load_data(self, test_nodes):
        # labels = None
        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):

            if self.is_toy == False:
                node_features = load_npz("../final_extraction/{}/{}_matrices/lsi_ngram.npz".format(self._dataset,self._dataset)).toarray()
                node_features = Normalizer().fit_transform(node_features)
                node_features = np.vstack((np.zeros(node_features.shape[-1]), node_features))

                # with tf.device('/cpu:0'):
                self.node_emb = tf.get_variable(name="node_emb", shape=node_features.shape, initializer=tf.constant_initializer(node_features), dtype=self.dtype,
                                                    trainable=False, use_resource=True)
                self.config['dim_Av'] = self.node_emb.get_shape().as_list()[-1]
                # labels = load_npz("../final_extraction/{}/{}_matrices/label.npz".format(self._dataset)).astype(np.int32)
                # labels = vstack((np.zeros(labels.shape[-1]), labels)).tocsr()

                if "delve" not in self._dataset:
                    try:
                        #edge_context_file = "../final_extraction/{}_ref_context.txt".format(self._dataset)
                        edge_features = load_npz("../final_extraction/{}/{}_matrices/lsi_c_ngram.npz".format(self._dataset,self._dataset)).toarray()
                        # self.empty_edge_att =  tf.get_variable("empty_edge_att",initializer=np.where(np.any(edge_features, -1) == False)[0])
                        edge_features = np.vstack((np.zeros(edge_features.shape[-1]), edge_features))
                        self.edge_features = Normalizer().fit_transform(edge_features)
                        # self.edge_features = np.hstack((self.edge_features, self.get_citecount(edge_context_file)))
                        # with tf.device('/cpu:0'):
                        self.edge_emb_init= tf.placeholder(tf.float32, shape=self.edge_features.shape)
                        self.edge_emb = tf.Variable(self.edge_emb_init, name="edge_emb", trainable=False, use_resource=True)
                        self.config['dim_Ae'] = edge_features.shape[-1]
                        self.has_edge_attr = True

                    except:
                        print("Error reading the reference file. Defaulting to text only")
                        self.edge_emb = None
                        self.config['dim_Ae'] = 0
                        self.has_edge_attr = False
                else:
                    self.edge_emb = None
                    self.config['dim_Ae'] = 0
                    self.has_edge_attr = False

            else:
                # features, labels, _split = load_cite_data(_dataset)
                # ref_features = None
                exit()


            self.Y = tf.placeholder(self.dtype, shape=(None, self.config['dim_y']))
            # self.edge_Weight = tf.get_variable("edge_weights",initializer=tf.ones([self.edge_emb.get_shape().as_list()[0]]), trainable=False)
            # self.edge_attention()
            self.build_helpers(test_nodes)
            # self.config['dim_Ae'] = self.edge_emb.get_shape().as_list()[-1]

            # for i in range(self.config['dim_y']):
            #     self.label_agents.append(rw.RAW(self.config['l_dim'], self.config['walk_len'], self.config['L'], i, self.node_emb, self.edge_emb, self.edges_per_node, self. self.is_training, self.get_path, self.config['dim_Av'], self.config['dim_Ae']))
            # self.htl = self.GRU(self.X)
            # self.htu = self.GRU(self.Xu)
            # cost function and optimizer

            # self.htl = self.GRU(self.X)
            # self.htu = self.GRU(self.Xu)
            # cost function and optimizer
            self.attention_node = tf.Variable(tf.ones([self.config['dim_y'],self.config['trainA'], self.batch_size]),validate_shape=False) #tf.get_variable("att_vect", shape=[self.config['trainA'], 128])
            self.attention_edge = tf.Variable(tf.ones([self.config['dim_y'], self.config['trainA'], self.batch_size]),validate_shape=False) #tf.get_variable("att_vect", shape=[self.config['trainA'], 128])

            self.cost = self.__cost(self.X)

            # self.ucost = self.__ucost(self.Xu )

            # classification accuracy
            # self.acc = self.__accuracy(self.X, self.Y)
            self.get_embd = self.get_embbeding(self.X)
            self.get_pt = self.__get_walk(self.X)
            self.pred = self.__predict(self.X)



        # return labels

    def __str__(self):
        '''
        report configurations
        '''
        msg = []
        for key in self.config:
            msg.append('{0:15}:{1:>20}\n'.format(key, self.config[key]))
        return '\n'.join(msg)

    def denseNDArrayToSparseTensor(self, arr):
        idx  = np.where(arr != 0.0)
        return tf.SparseTensor(np.vstack(idx).T, arr[idx], arr.shape)

    def build_helpers(self, test_nodes):
        try:
            edgelist = open("../final_extraction/{}/{}_weighted.edgelist".format(self._dataset,self._dataset), "rU").read().split("\n")[:-1]
        except:
            self.train_edge_weigths_cosine()
            edgelist = open("../final_extraction/{}/{}_weighted.edgelist".format(self._dataset,self._dataset), "rU").read().split("\n")[:-1]


        neighbor = {}
        edge_tensor = [0,0]
        iter = 2
        # edge_weight = []
        neighbor[0] = [[], [[0, 1]]]
        neighbor[0] = [[[0, 1]], []]
        for edge in edgelist:
            edgei = edge.split('\t')
            s, t = map(int, edgei[:2])

            s, t = s+1, t+1
            w = float(edgei[2])
            # edge_weight.append(w)

            if neighbor.has_key(t):
                neighbor[t][1].append([iter,w])
            else:
                neighbor[t] = [[],[[iter,w]]]

            iter += 1
            if neighbor.has_key(s):
                neighbor[s][0].append([iter, w])
            else:
                neighbor[s] = [[[iter, w]], []]


            edge_tensor.extend((s,t))
            iter += 1

        # edges_per_node = np.zeros((len(neighbor), self.config['L']))
        # for key, value in neighbor.iteritems():
        #     value[0] = np.array(value[0])
        #     value[1] = np.array(value[1])
        #     half = int(self.config['L'] / 2)
        #     if value[0].shape[0] > 0:
        #         if value[0].shape[0] <= half:
        #             edges_per_node[key, :value[0].shape[0]] = value[0][:, 0]
        #             space = self.config['L'] - value[0].shape[0]
        #             if value[1].shape[0] > 0:
        #                 others = value[1][:, 0]
        #                 others = others[np.argsort(value[1][:, 1])[::-1]]
        #                 if others.shape[0] >= half:# - value[0].shape[0]):
        #                     edges_per_node[key, value[0].shape[0]:value[0].shape[0] + others.shape[0]] = others[:space]
        #                 else:
        #                     edges_per_node[key, value[0].shape[0]:value[0].shape[0] + others.shape[0]] = others
        #             # edges_per_node[key, -1] = 1
        #         else:
        #             rank = np.argsort(value[0][:, 1])[::-1]
        #             edges_per_node[key, :half] = value[0][rank[:half], 0]
        #             if value[1].shape[0] < 1:
        #                 edges_per_node[key, half:rank.shape[0]] = value[0][rank[half: self.config['L']] , 0]
        #             else:
        #                 others = value[1][:, 0]
        #                 others = others[np.argsort(value[1][:, 1])[::-1]]
        #                 if others.shape[0] >= half:  # - value[0].shape[0]):
        #                     edges_per_node[key, half:] = others[:half]
        #                 else:
        #                     edges_per_node[key, half:(half + others.shape[0])] = others
        #                     space = self.config['L'] - (half + others.shape[0])
        #                     space = (rank.shape[0] - half) if rank.shape[0] <(half+ space) else space
        #                     edges_per_node[key, (half + others.shape[0]):(half + others.shape[0])+space] = value[0][rank[half: (half + space)], 0]
        #             # edges_per_node[key, -1] = 1
        #     elif value[1].shape[0] > 0:
        #         others = value[1][:, 0]
        #         others = others[np.argsort(value[1][:, 1])[::-1]]
        #         if others.shape[0] >= self.config['L']:
        #             edges_per_node[key, :] = others[:self.config['L']]
        #         else:
        #             edges_per_node[key, :others.shape[0]] = others

        edges_per_node = np.zeros((len(neighbor), self.config['L']))
        for key, value in neighbor.iteritems():
            value[0] = np.array(value[0])
            value[1] = np.array(value[1])
            half = int(self.config['L'] / 2)
            if value[0].shape[0] > 0:
                if value[0].shape[0] <= half:
                    edges_per_node[key, :value[0].shape[0]] = value[0][:, 0]
                    space = self.config['L'] - value[0].shape[0]
                    if value[1].shape[0] > 0:
                        others = value[1][:, 0]
                        # others = others[np.argsort(value[1][:, 1])[::-1]]
                        if others.shape[0] >= space:# - value[0].shape[0]):
                            others_samp = rn.sample(others, space)
                            edges_per_node[key, value[0].shape[0]:value[0].shape[0] + space] = others_samp
                        else:
                            edges_per_node[key, value[0].shape[0]:value[0].shape[0] + others.shape[0]] = others
                    # edges_per_node[key, -1] = 1
                else:
                    rank = value[0][:, 0]
                    samp = rn.sample(rank, half) #rank[:half]
                    cur = list(set(rank).difference(samp))
                    edges_per_node[key, :half] = samp
                    if value[1].shape[0] < 1:
                        edges_per_node[key, half:half+len(cur)] = cur[:half]
                    else:
                        others = value[1][:, 0]
                        # others = others[np.argsort(value[1][:, 1])[::-1]]
                        # others_rank = np.argsort(value[1][:, 1])[::-1]
                        if others.shape[0] >= half:  # - value[0].shape[0]):
                            others_samp = rn.sample(others, half)
                            # others_cur = set(others).difference(others_samp)
                            edges_per_node[key, half:] = others_samp
                        else:
                            edges_per_node[key, half:(half + others.shape[0])] = others
                            space = self.config['L'] - (half + others.shape[0])
                            space = len(cur) if len(cur) < space else space
                            edges_per_node[key, (half + others.shape[0]):(half + others.shape[0])+space] = cur[:space]
                    # edges_per_node[key, -1] = 1
            elif value[1].shape[0] > 0:
                others = value[1][:, 0]
                # others = others[np.argsort(value[1][:, 1])[::-1]]
                if others.shape[0] >= self.config['L']:
                    others_samp = rn.sample(others, self.config['L'])
                    edges_per_node[key, :] = others_samp
                else:
                    edges_per_node[key, :others.shape[0]] = others
        # with tf.device('/cpu:0'):
        if self.config['is_transductive']:
            test_mask = edges_per_node > 0
        else:
            test_mask = ~np.isin(np.array(edge_tensor)[edges_per_node.astype(np.int32)], test_nodes)
            null_neighboors = edges_per_node > 0
            test_mask = np.logical_and(test_mask, null_neighboors)
        self.edges_per_node = tf.convert_to_tensor(edges_per_node, tf.int64)#self.denseNDArrayToSparseTensor(edges_per_node)
        self.edge_tensor = tf.convert_to_tensor(edge_tensor, tf.int64)
        self.test_mask = tf.convert_to_tensor(test_mask, tf.bool)

                #
                # if value[1].shape[0] <= self.config['L']:
                #     edges_per_node[key, :len(value[1])] = value[1]
                #     edges_per_node[key, -1] = 0
                # else:
                #     edges_per_node[key, :self.config['L']] = value[1][:self.config['L']]
                #     edges_per_node[key, -1] = 0
        # self.neigh_count = tf.convert_to_tensor(np.count_nonzero(edges_per_node, -1), dtype=self.dtype)
        # with tf.device('/cpu:0'):
        # self.edges_per_node = tf.convert_to_tensor(edges_per_node, tf.int32)#self.denseNDArrayToSparseTensor(edges_per_node)
        # self.edge_tensor = tf.convert_to_tensor(edge_tensor, tf.int32)


    def train_test_split(self, y, ratio):
        label = y.toarray()
        labeled_set = np.where(label.sum(-1) > 0)[0]
        train_mask = np.array(rn.sample(labeled_set, int(len(labeled_set) * ratio)))
        cur = set(labeled_set).difference(train_mask)
        val_mask = np.array(rn.sample(cur, np.clip(int(len(labeled_set) * 0.1), 100, 2000)))
        # val_mask = np.array([])
        # for i in range(label.shape[1]):
        #     sel = np.where(label[train_mask,i] > 0)[0]
        #     val_mask = np.hstack((val_mask, train_mask[np.random.choice(sel, int(np.ceil(0.1*sel.shape[0])))]))
        test_mask = np.array(list(set(cur).difference(val_mask)))
        # train_mask = np.array(list(set(train_mask).difference(val_mask)))
        # u_mask = np.setdiff1d(np.arange(label.shape[0]),train_mask)#np.where(label.sum(-1) <= 0)[0]

        return train_mask, val_mask, test_mask #, u_mask





    def edge_attention(self, reuse=tf.AUTO_REUSE):

        edge_Weight = tf.layers.dense(self.edge_emb, self.config['l_dim'], name='e_e_dense1',activation=self.config['activation'], reuse=reuse)
        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            att = tf.get_variable("att").assign(tf.squeeze(tf.layers.dense(edge_Weight, 1, name='e_e_dense2_{}'.format(lid),activation=tf.nn.sigmoid, reuse=reuse, trainable=False),-1))
            att = tf.scatter_update(att,self.empty_edge_att,0.0*tf.ones(self.empty_edge_att.get_shape()))
            att = tf.scatter_update(att, tf.Variable([0]), tf.Variable([0.0]))  #zero padding index
            self.edge_Weight = self.edge_Weight.assign(att)

    def gather_cols(self, params, indices):
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1], [-1]) + indices, [-1])
        partitions = tf.reduce_sum(tf.one_hot(i_flat, tf.shape(p_flat)[0], dtype='int32'), 0)
        col_sel = tf.dynamic_partition(p_flat, partitions, 2)  # (batch_size, n_dim)
        col_sel = col_sel[1]
        return tf.reshape(col_sel,
                          [p_shape[0], 1])

    def gather_colsv2(self, params, indices):
        p_shape = tf.shape(params)
        A = params.get_shape().as_list()[:2]
        p_flat = tf.reshape(params, [-1])
        indices = tf.reshape(indices, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[1]*p_shape[0]*p_shape[2]) * p_shape[3], [-1]) + indices, [-1])
        partitions = tf.reduce_sum(tf.one_hot(i_flat, tf.shape(p_flat)[0], dtype='int32'), 0)
        col_sel = tf.dynamic_partition(p_flat, partitions, 2)  # (batch_size, n_dim)
        col_sel = col_sel[1]
        return tf.reshape(col_sel,[A[0],A[1], p_shape[2], 1])


    def index_along_every_row(self, array, index):
        N, _ = array.shape
        ids =  array[np.arange(N), index]
        return ids



    def sample_neighbor(self, current_x):
        neighbors = tf.gather(self.edges_per_node, current_x)
        neighbors_weight = tf.gather(self.edge_Weight, neighbors)  #get the weights of the connecting edges
        # neighbors_weight = tf.nn.embedding_lookup(self.edge_Weight, neighbors[:, :-1])

        # neighbors_weight = neighbors_weight / tf.reduce_sum(neighbors_weight)
        # next_id, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        #                                                        true_classes = neighbors,
        #                                                        num_true = self.config['walk_len'],
        #                                                        num_sampled = 1,
        #                                                        unique = False,
        #                                                        range_max = self.config['walk_len'],
        #                                                        unigrams = neighbors_weight # this is P, times 100
        #                                                     )
        next_id_sample = tf.distributions.Categorical(probs=neighbors_weight).sample()
        next_id = self.gather_cols(neighbors, next_id_sample)
        # next_id = tf.py_func(self.index_along_every_row, [neighbors, next_id_sample], [tf.int32])[0]
        # next_id =tf.reshape(next_id, [-1,1])
        # zip row indices with column indices
        # full_indices = tf.stack([next_id, tf.expand_dims(neighbors[:, -1],-1)], axis=-1)
        next_id= tf.gather(self.edge_tensor, next_id)
        # next_id =tf.reshape(current_x, [-1,1])

        # next_id = neighbors_weight[:,3:4]

        mask = tf.greater_equal(neighbors_weight, 0.5)
        neighbors_selection =  tf.multiply(neighbors, tf.cast(mask, tf.int64)) #neighbors * mask
        neighbor_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.edge_emb, neighbors_selection), 1)

        #TODO: Add zeros to begining of edge embed
        return next_id, neighbor_emb

    def sample_random_neighbor(self, current_x):
        neighbors = tf.gather(self.edges_per_node, current_x)
        mask = tf.cast(tf.greater(neighbors, 0), tf.float32)
        neighbors_weight = tf.divide(mask, tf.reduce_sum(mask))
        id_sample = tf.distributions.Categorical(probs=neighbors_weight).sample()
        samp_id = self.gather_colsv2(neighbors, id_sample)
        # def condition(index, samp_id):
        #     return tf.less(index, self.config['A'])
        #
        # def body(index, samp_id):
        #     samp_id = tf.concat((samp_id,tf.expand_dims(self.gather_cols(neighbors[index, :,:], id_sample[index, :]), 0)), 0)
        #     return tf.add(index, 1), samp_id
        #
        # samp_id = tf.expand_dims(self.gather_cols(neighbors[0,:,:], id_sample[0, :]),0)
        # index_loop = (tf.constant(1), samp_id)
        # samp_id = tf.while_loop(condition, body, index_loop, parallel_iterations=10,
        #                   shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, None, 1])))[-1]
        # # next_id = tf.py_func(self.index_along_every_row, [neighbors, next_id_sample], [tf.int32])[0]
        # samp_id =tf.reshape(samp_id, [self.config['A'], -1,1])


        # samp_id = self.gather_cols(neighbors, id_sample)
        samp_id = tf.gather(self.edge_tensor, samp_id)

        return tf.squeeze(samp_id,-1)



    def dense(self, inputs, output_dim, name):
        shape = inputs.get_shape().as_list()
        if len(shape) > 4:
            W = tf.get_variable(name='W_{}'.format(name) ,
                                initializer = lambda: tf.glorot_uniform_initializer()(( shape[0], shape[-1], output_dim)))
            # P = tf.get_variable(name='P_{}'.format(name) ,
                                # initializer = lambda: tf.glorot_uniform_initializer()((shape[0], 1, 1, 1, 1)))

            b = tf.get_variable(name='b_{}'.format(name) , initializer = lambda: -1*tf.ones_initializer()(output_dim))

            # return tf.nn.bias_add(tf.multiply(tf.tensordot(inputs, W, [[-1], [0]]), P), b)
            return tf.nn.bias_add(tf.einsum('abilj,ajk->abilk', inputs, W), b)
        else:
            W = tf.get_variable(name='W_{}'.format(name) ,
                                initializer = lambda: tf.glorot_uniform_initializer()((shape[0],  shape[-1], output_dim)))
            # P = tf.get_variable(name='P_{}'.format(name) ,
                                # initializer = lambda: tf.glorot_uniform_initializer()((shape[0], 1, 1, 1)))

            b = tf.get_variable(name='b_{}'.format(name) , initializer = lambda: -1*tf.ones_initializer()(output_dim))

            # return tf.nn.bias_add(tf.multiply(tf.tensordot(inputs, W, [[-1], [0]]), P), b)
            return tf.nn.bias_add(tf.einsum('abij,ajk->abik', inputs, W), b)


    def sample_neighbor_walk(self, current_x, current_emb, h, t, reuse=tf.AUTO_REUSE):
        neighbors = tf.gather(self.edges_per_node, current_x)
        mask_neighbors = tf.gather(self.test_mask, current_x)
        # mask = tf.greater(neighbors,0 )
        mask = tf.cond(self.is_training, lambda : mask_neighbors, lambda : tf.greater(neighbors,0 ))
        h = tf.tile(tf.expand_dims(h, 3), [1,1, 1, self.config['L'], 1])
        neighbor_node_emb = tf.nn.embedding_lookup(self.node_emb, tf.gather(self.edge_tensor, neighbors))
        current_emb = tf.tile(tf.expand_dims(current_emb, 3), [1,1,1, self.config['L'], 1])
        if self.has_edge_attr:
            neighbor_edge_emb = tf.nn.embedding_lookup(self.edge_emb,tf.div(neighbors,2))
            neighbor_emb_act = tf.add_n((neighbor_edge_emb, current_emb, neighbor_node_emb))
        else:
            neighbor_emb_act = tf.add_n((current_emb, neighbor_node_emb))

        att_emb = tf.concat((h, neighbor_emb_act), -1)
        # att_emb = tf.multiply(att_emb,tf.cast(tf.expand_dims(mask, -1), tf.float32))

        # neighbor_node_emb = tf.nn.embedding_lookup(self.node_emb, tf.gather(self.edge_tensor, neighbors))

        neighbors_weight = tf.squeeze(tf.keras.backend.hard_sigmoid(self.dense(att_emb, 1, name='e_e_dense2')),-1)
        neighbors_weight = tf.multiply(neighbors_weight,tf.cast(mask, tf.float32))

        filter_neighbors = tf.greater_equal(neighbors_weight, 0.5)
        mask2 = tf.logical_and(filter_neighbors, mask)
        # neighbors_weight = tf.multiply(neighbors_weight, tf.cast(mask2, tf.float32))
        neighbors_weight = tf.div_no_nan(neighbors_weight , tf.reduce_sum(neighbors_weight))
        # next_id_sample = tf.distributions.Categorical(probs=neighbors_weight).sample()
        next_id_sample = tf.expand_dims(tf.distributions.Categorical(probs=neighbors_weight).sample(), -1)

        # next_id = self.gather_colsv2(neighbors, next_id_sample)
        next_id = tf.batch_gather(neighbors, next_id_sample)

        next_id= tf.nn.embedding_lookup(self.edge_tensor, next_id)

        neighbor_emb = tf.reduce_sum(tf.multiply(neighbor_node_emb, tf.expand_dims(tf.cast(mask2, tf.float32), -1)),3)
        # is_sample_masked = self.gather_colsv2(mask, next_id_sample)
        is_sample_masked = tf.batch_gather(mask, next_id_sample)
        non_isolated_nodes = tf.logical_and(tf.reduce_any(mask, -1), tf.squeeze(is_sample_masked,-1))

        next_id = tf.add(tf.multiply(tf.squeeze(next_id,-1),tf.cast(non_isolated_nodes, tf.int64)) , tf.multiply(current_x,tf.cast(~non_isolated_nodes, tf.int64)))

        # likelihood = tf.squeeze(self.gather_colsv2(neighbors_weight, next_id_sample),[-1])#tf.add(tf.squeeze(self.gather_colsv2(neighbors_weight, next_id_sample),[-1]) , tf.cast(~non_isolated_nodes, tf.float32))
        likelihood = tf.squeeze(tf.batch_gather(neighbors_weight, next_id_sample),[-1])
        likelihood = tf.multiply(tf.pow(self.config['gamma'], (self.T - t)), tf.ones_like(tf.log(tf.clip_by_value(likelihood,1e-10,1.0))))

        return tf.expand_dims(next_id,-1), neighbor_emb, tf.expand_dims(likelihood,-1)



    def action_walk_edge(self, current_x, h, t, reuse=tf.AUTO_REUSE):
        neighbors = tf.gather(self.edges_per_node, current_x)
        mask = tf.greater(neighbors,0 )
        neighbor_edge_emb = tf.nn.embedding_lookup(self.edge_emb,tf.div(neighbors,2))
        h = tf.tile(tf.expand_dims(h, 3), [1,1,1, self.config['L'], 1])
        # is_valid_edge = tf.equal(tf.reduce_sum(neighbor_emb, -1), 0)
        att_emb = tf.concat((h, neighbor_edge_emb), -1)
        neighbors_weight = tf.squeeze(tf.keras.backend.hard_sigmoid(self.dense(att_emb, 1, "policy_edge")),-1)
        neighbors_weight = tf.multiply(neighbors_weight,tf.cast(mask, tf.float32))

        neighbors_weight = tf.divide(neighbors_weight , tf.expand_dims(tf.reduce_sum(neighbors_weight,-1), -1))
        next_id_sample = tf.distributions.Categorical(probs=neighbors_weight).sample()

        next_id = self.gather_colsv2(neighbors, next_id_sample)
        next_id= tf.nn.embedding_lookup(self.edge_tensor, next_id)

        current_emb = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.one_hot(next_id_sample, 40), -1), neighbor_edge_emb), -2)

        op = tf.assign_add(self.attention_edge, tf.multiply(tf.pow(self.config['gamma'], (self.T - t)), tf.log(tf.clip_by_value(tf.squeeze(self.gather_colsv2(neighbors_weight, next_id_sample),[-1]),1e-10,1.0))))

        #TODO: Add zeros to begining of edge embed
        with tf.control_dependencies([op]):
            return next_id, current_emb




    def GRU(self, trueX):

        def forward(input, t):
            """Perform a forward pass.

            Arguments
            ---------
            h_tm1: np.matrix
                The hidden state at the previous timestep (h_{t-1}).
            x_t: np.matrix
                The input vector.
            """
            # Definitions of z_t and r_t

            h_tm1 = input[:,:,:,:self.config['l_dim']]
            x = tf.cast(input[:,:,:,self.config['l_dim']], tf.int64)
            h_tm1 = tf.cond(self.is_training, lambda : h_tm1 * self.dropout_recurrent, lambda : h_tm1)


            x_t = tf.nn.embedding_lookup(self.node_emb, x)
            next_x, x_t_e, likelihood = self.sample_neighbor_walk(x, x_t, h_tm1, t)
            x_t = tf.concat([x_t, x_t_e], -1)
            #
            # if is_edge:
            #     next_x, x_t = self.action_walk_edge( x, h_tm1, t)
            # else:
            #     x_t = tf.nn.embedding_lookup(self.node_emb, x)
            #     next_x = self.action_walk(x, h_tm1, t)


            # if is_edge:
            #     zr_t = tf.keras.backend.hard_sigmoid(self.dense(tf.concat([x_t, h_tm1],-1), self.config['l_dim']*2, 'zr_e'))
            #     z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)
            #     r_state = r_t * h_tm1
            #     h_proposal = tf.tanh(self.dense(tf.concat([x_t, r_state],-1), self.config['l_dim'], 'h_e'))
            # else:
            #     zr_t = tf.keras.backend.hard_sigmoid(self.dense(tf.concat([x_t, h_tm1],-1), self.config['l_dim']*2, 'zr'))
            #     z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)
            #     r_state = r_t * h_tm1
            #     h_proposal = tf.tanh(self.dense(tf.concat([x_t, r_state],-1), self.config['l_dim'], 'h'))

            zr_t = tf.keras.backend.hard_sigmoid(self.dense(tf.concat([x_t, h_tm1],-1), self.config['l_dim']*2, name='zr'))
            z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)
            r_state = r_t * h_tm1
            h_proposal = tf.tanh(self.dense(tf.concat([x_t, r_state],-1), self.config['l_dim'], name='h'))


            # Compute the next hidden state
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

            # return tf.concat([h_t, tf.cast(next_x, self.dtype), x_t],-1)
            return tf.concat([h_t, tf.cast(next_x, self.dtype), likelihood, x_t],-1)

        # # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
        dummy_emb = tf.tile(tf.expand_dims(tf.cast(trueX, self.dtype),-1), [1,1,1,self.config['dim_Av']])
        # dummy_emb = tf.nn.embedding_lookup(self.node_emb, trueX)
        # h_0 = tf.einsum('abij,k->abik', dummy_emb, tf.zeros([self.config['l_dim']],dtype=self.dtype))
        shape = dummy_emb.get_shape().as_list()
        h_0 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(shape[0],shape[1], self.config['dim_Av'], self.config['l_dim'])),
                        name='h_0' )
        next_x0 = tf.expand_dims(tf.cast(trueX, self.dtype),-1)
        concat_tensor = tf.concat([h_0, next_x0, next_x0, dummy_emb, dummy_emb], -1)

        if self.is_train == True:
            dropout_hidden = tf.nn.dropout(h_0[0,0,:,:], 0.80, name='dropout2')
        self.dropout_recurrent = tf.get_default_graph().get_tensor_by_name('{}/Floor:0'.format("/".join(dropout_hidden.name.split("/")[:-1])))

        #
        # h_0 = tf.matmul(cur_emb, tf.zeros(dtype=tf.float32, shape=(shape[0],shape[1], self.config['dim_Av'], self.config['l_dim'])),
        #                 name='h_0' )
        # concat_tensor = tf.concat([h_0, next_x0, cur_emb], -1)

        h_t = tf.scan(forward, self.range, initializer = concat_tensor,parallel_iterations=20,
                      name='h_t_transposed' )

        # Transpose the result back
        # h_t = tf.transpose(h_t, [1, 0, 2], name='h_t' )
        h_t_b = self.BGRU(tf.reverse(h_t[:,:,:,:,self.config['l_dim'] +2:], [0]))
        # ht =  tf.concat((h_t[-1,:,:self.config['l_dim']] , h_t_b[-1,:,:]),-1)
        ht =  h_t[-1,:,:,:,:self.config['l_dim']] + h_t_b
        output = tf.cond(self.get_path, lambda : tf.transpose(h_t[:,:,:, :,self.config['l_dim']], perm=[3,1,2,0]), lambda : ht)
        return output, tf.reduce_sum(h_t[:-1,:, :,:,self.config['l_dim']+1],0)



    def BGRU(self, node_emb):


        def backward(h_tm1, x_t):
            """Perform a forward pass.

            Arguments
            ---------
            h_tm1: np.matrix
                The hidden state at the previous timestep (h_{t-1}).
            x_t: np.matrix
                The input vector.
            """


            h_tm1 = tf.cond(self.is_training, lambda: h_tm1 * self.dropout_recurrent_b, lambda: h_tm1)
            # h_tm1 *= self.dropout_recurrent_b
            # x_t *= self.dropout_b
            # Definitions of z_t and r_t
            # shape = x_t.get_shape().as_list()

            zr_t = tf.keras.backend.hard_sigmoid(self.dense(tf.concat([x_t, h_tm1],-1), self.config['l_dim']*2, name ='zr_b'))
            z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)
            r_state = r_t * h_tm1
            h_proposal = tf.tanh(self.dense(tf.concat([x_t, r_state],-1), self.config['l_dim'], name='h_b'))
            # # Definition of h~_t
            # h_proposal = tf.tanh(tf.nn.bias_add(tf.tensordot(x_t, Wh_b, [[len(shape) - 1], [0]]) + tf.tensordot(tf.multiply(r_t, h_tm1), Uh_b, [[len(shape) - 1], [0]]) , bh_b))

            # Compute the next hidden state
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

            return h_t



        # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
        # h_0_b = tf.einsum('abij,k->abik', node_emb[0, :, :, :, :], tf.zeros([self.config['l_dim']],dtype=self.dtype))
        shape = node_emb.get_shape().as_list()
        h_0_b = tf.matmul(node_emb[0, :, :, :, :], tf.zeros(dtype=tf.float32, shape=(shape[1],shape[2], self.config['dim_Av']*2, self.config['l_dim'])),
                          name='h_0_b' )

        # dropout_input = tf.nn.dropout(node_emb[0, :, :], 0.5, name='dropout1b' )
        # self.dropout_b = tf.get_default_graph().get_tensor_by_name('cost/dropout1b/Floor:0' )
        if self.is_train == True:
            dropout_hidden = tf.nn.dropout(h_0_b[0,:,:], 0.80, name='dropout2b' )
            self.dropout_recurrent_b = tf.get_default_graph().get_tensor_by_name('{}/Floor:0'.format("/".join(dropout_hidden.name.split("/")[:-1])))

        h_t_transposed_b = tf.scan(backward, node_emb, initializer = h_0_b,parallel_iterations=20, name='h_t_transposed_b' )

        # shape = node_emb.get_shape().as_list()
        # h_0_b = tf.matmul(node_emb[0, :, :, :, :], tf.zeros(dtype=tf.float32, shape=(shape[1],shape[2], self.config['dim_Av'], self.config['l_dim'])),
        #                   name='h_0_b' )
        #
        # # dropout_input = tf.nn.dropout(node_emb[0, :, :], 0.5, name='dropout1b' )
        # # self.dropout_b = tf.get_default_graph().get_tensor_by_name('cost/dropout1b/Floor:0' )
        # if self.is_train == True:
        #     dropout_hidden = tf.nn.dropout(h_0_b[0,:,:], 0.15, name='dropout2b' )
        #     self.dropout_recurrent_b = tf.get_default_graph().get_tensor_by_name('{}/Floor:0'.format("/".join(dropout_hidden.name.split("/")[:-1])))
        #
        # # Perform the scan operator
        # h_t_transposed_b = tf.scan(backward, node_emb, initializer = h_0_b, name='h_t_transposed_b' )

        # Transpose the result back
        # self.h_t_b = tf.transpose(self.h_t_transposed_b, [1, 0, 2], name='h_t_b' )
        return h_t_transposed_b[-1,:,:,:,:]





    def __cost(self, trueX):
        '''
        compute the cost tensor

        trueX -- input X (2D tensor)
        trueY -- input Y (2D tensor)
        reuse -- whether to reuse the NNs

        return 1D tensor of cost (batch_size)
        '''

        # Y_columns = tf.unstack(self.Y)




        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):


            X = tf.expand_dims(tf.expand_dims(trueX, 0),0 )
            X = tf.tile(X, [self.config['dim_y'], self.config['trainA'],1])

            # def body (_cost, index):
            Y = tf.transpose(tf.expand_dims(self.Y,-1), perm=[1,2,0])
            Y = tf.tile(Y, [1,self.config['trainA'], 1])

            Z, likelihood = self.GRU(X)
            Z = tf.reshape(Z, [self.config['dim_y'], self.config['trainA'], -1,self.config['l_dim']])
            # Z = tf.layers.dense(Z, self.config['l_dim'], name='Z2Z', activation=self.config['activation'], kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01), reuse=tf.AUTO_REUSE)

            # Z = Z_e + Z_n
            log_pred_Y = tf.squeeze(self.dense(Z, 1, name='Z2y'),-1)

            reward = tf.cast(tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(log_pred_Y), 0.5), self.dtype), Y), tf.float32)
            reward = 2*(reward-0.5)

            # use the classification error as regularizer

            _cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(Y, self.dtype), logits=log_pred_Y)

            # _cost = tf.reduce_mean(i_cost) - tf.reduce_mean(reward * self.attention_node) - tf.reduce_mean(reward * self.attention_edge)

            # return _cost
            return tf.reduce_mean(_cost) - tf.reduce_mean(reward * likelihood)


    def __classify(self, trueX, reuse=tf.AUTO_REUSE):
        '''
        classify input 2D tensor

        return 1D tensor (integer class labels)
        '''
        with tf.variable_scope("cost", reuse=True):


            # X -> Z
            X = tf.expand_dims(tf.expand_dims(trueX, 0),0)
            X = tf.tile(X, [self.config['dim_y'], self.config['testA'],1])
            Z, likelihood = self.GRU(X)
            Z = tf.reshape(Z, [self.config['dim_y'], self.config['trainA'], -1,self.config['l_dim']])
            # Z = tf.layers.dense(Z, self.config['l_dim'], name='Z2Z', activation=self.config['activation'], kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01), reuse=tf.AUTO_REUSE)

            log_pred_Y = tf.squeeze(self.dense(Z, 1, name='Z2y'),-1)
            Y = tf.reduce_mean(tf.nn.sigmoid(log_pred_Y),1)
            Y = tf.cast(tf.greater_equal(Y, 0.5), tf.int32)

            return tf.transpose(Y)


    def get_embbeding(self, trueX, reuse=tf.AUTO_REUSE):
        '''
        classify input 2D tensor

        return 1D tensor (integer class labels)
        '''
        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):

            # X -> Z
            X = tf.expand_dims(tf.expand_dims(trueX, 0),0)
            X = tf.tile(X, [self.config['dim_y'], self.config['testA'],1])
            Z, _ = self.GRU(X)
            Z = tf.reshape(Z, [self.config['dim_y'], self.config['trainA'], -1,self.config['l_dim']])

            # Y = tf.reduce_mean(Y, 0)
            return Z


    def __get_walk(self, trueX, reuse=tf.AUTO_REUSE):
        '''
        classify input 2D tensor

        return 1D tensor (integer class labels)
        '''
        X = tf.expand_dims(tf.expand_dims(trueX, 0),0)
        X = tf.tile(X, [self.config['dim_y'], self.config['testA'],1])
        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            walk, _ = self.GRU(X)
            walk = tf.reshape(walk, [-1,self.config['dim_y'],self.config['trainA'], self.config['walk_len']])

            # if "delve" not in self._dataset:
            #     return [tf.reshape(self.GRU(X, False), [-1,self.config['dim_y'],self.config['trainA'], self.config['walk_len']]), tf.reshape(self.GRU(X, True), [-1,self.config['dim_y'],self.config['trainA'], self.config['walk_len']])]
            # else:
            #     return [tf.reshape(self.GRU(X, False), [-1,self.config['dim_y'],self.config['trainA'], self.config['walk_len']])]
            return walk


    def __accuracy(self, trueX, trueY):
        '''
        measure the semi-supervised accuracy

        trueX -- input 2D tensor (features)
        trueY -- input 2D tensor (one-hot labels)

        return a scalar tensor
        '''

        eval_results = []
        # non_zero = tf.count_nonzero(predicted)

        predicted = self.__classify(trueX)
        actual = tf.argmax(trueY, axis=-1)

        corr = tf.equal(predicted, actual)
        acc = tf.reduce_mean(tf.cast(corr, tf.float32))

        return acc

    def fetch_batches(self, allidx, nbatches, batchsize, wind=True):
        '''
        allidx  -- 1D array of integers
        nbatches  -- #batches
        batchsize -- mini-batch size

        split allidx into batches, each of size batchsize
        '''
        N = allidx.size

        for i in range(nbatches):
            if wind:
                idx = [(_ % N) for _ in range(i * batchsize, (i + 1) * batchsize)]
            else:
                idx = [_ for _ in range(i * batchsize, (i + 1) * batchsize) if (_ < N)]

            yield allidx[idx]

    def early_stop_criteria(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.steps_no_increase = 0
            # if not os.path.exists(self.save_path):
            #     os.makedirs(self.save_path)
            #
            # saver.save(sess, '{}/model'.format(self.save_path))
        else:
            self.steps_no_increase += 1
            if self.steps_no_increase == self.patience:
                return True
        return False

    def train(self, labels, train_ratio, masks, nepochs=100, verbose=True, save=False):

        train_mask, test_mask = masks
        if save == False:
            # N = len(train_mask)
            allidx = np.arange(train_mask.shape[0])
            Y_train = labels[train_mask, :]
            Y_test = labels[test_mask, :]
            nbatches = int(np.ceil(train_mask.shape[0] / self.batch_size))
        else:
            train_mask = np.hstack((train_mask, test_mask))
            # N = len(train_mask)
            save_npz('paths/{}_idx'.format(self.id), csr_matrix(train_mask))
            allidx = np.arange(train_mask.shape[0])
            Y_train = labels[train_mask, :]
            nbatches = int(np.ceil(train_mask.shape[0] / self.batch_size))


        self.is_train = True





        # cost = self.cost #+ (0.1 * self.ucost)



        # self.ucost = self.__ucost(self.Xu )

        #
        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            train_op = tf.train.AdamOptimizer(self.config['lrate']).minimize(self.cost)
            # train_op = tf.train.GradientDescentOptimizer(self.config['lrate']).minimize(self.cost)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if train_ratio == 1:
            saver = tf.train.Saver()
        glob_init = tf.global_variables_initializer()

        with tf.Session(config=config) as sess:

            # fp = open("tf_variables.txt", "w")
            # for op in tf.get_default_graph().get_operations():
            #     fp.write(str(op.name)+"\n")
            # exit()

            # with tf.device('/cpu:0'):
            if self.has_edge_attr:
                sess.run(glob_init, feed_dict={self.edge_emb_init: self.edge_features})
            else:
                sess.run(glob_init)
            sess.run(tf.local_variables_initializer())
            sess.graph.finalize()  # Graph is read-only after this statement.



            for e in range(nepochs):
                self.rng.shuffle(allidx)
                # self.rng.shuffle(allidxu)
                # self.edge_attention()


                epoch_cost = 0
                # verbose = True

                for batchl in self.fetch_batches(allidx, nbatches, self.batch_size):

                    if verbose:
                        epoch_cost += sess.run([train_op, self.cost],
                                               feed_dict={self.X: train_mask[batchl],
                                                          # self.Xu: u_mask[batchu],
                                                          self.Y: Y_train[batchl, :].toarray(),
                                                          self.is_training: False,
                                                          self.get_path: False})[1]
                    else:
                        sess.run(train_op,
                                 feed_dict={self.X: train_mask[batchl],
                                            # self.Xu: u_mask[batchu],
                                            self.Y: Y_train[batchl, :].toarray(),
                                            self.is_training: False,
                                            self.get_path: False})
                epoch_cost /= nbatches
                # predY = []  # np.zeros(Y.shape)

                # allidx = np.arange(len(val_mask))
                # nbatches = int(np.ceil(len(val_mask) / batchsize))
                #
                if verbose:
                    cost_only = True
                    if cost_only:
                        print('[{0:5d}] E={1:.8f}\n'.format(e, epoch_cost))
                    else:

                        # #output after each epoch
                        train_acc = sess.run(self.acc,
                                             feed_dict={self.X: train_mask[0:self.batch_size],
                                                        self.Y: Y_train[0:self.batch_size, :].toarray(),
                                                        self.is_training: False,
                                                        self.get_path: False})

                        print( '[{0:5d}]  TrainE={1:.4f}  TrainAcc={2:.4f}'.format( e, epoch_cost, train_acc ))

                # # pdb.set_trace()
                # if train_ratio < 1 and cv_splits==None:
                #     val_cost = 0
                #     for batchl in self.fetch_batches(allidx_val, nbatches_val, self.batch_size):
                #         val_cost += sess.run(self.cost,
                #                              feed_dict={self.X: val_mask[batchl],
                #                                         self.Y: Y_val[batchl, :].toarray(),
                #                                         self.is_training: False,
                #                                         self.get_path: False})
                #     val_cost /= nbatches_val
                #
                #     # pdb.set_trace()
                #     if (e > 2):
                #         to_stop = self.early_stop_criteria(val_cost)
                #         if to_stop == True:
                #             break
            # save the trained model
            # save the trained model

            # for batch in self.fetch_batches(allidx, nbatches, batchsize, wind=False):
            # with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            #     pred_Y = tf.get_variable("predY_{}".format(len(batch)), shape=[len(batch), self.config['dim_y']], initializer=tf.ones_initializer)
            #     sess.run(pred_Y.initializer)
            # test = sess.run([self.__get_walk(self.X)],
            #                 feed_dict={self.X: np.array([1]),
            #                            self.is_training: False,
            #                              self.get_path: True})[0]
            # print(test.astype(np.int32))
            # exit()
            # pdb.set_trace()

            if save:
                # save_path = '{}{}'.format(self.model_dir, self.id)
                # if not os.path.exists(self.save_path):
                #     os.makedirs(self.save_path)

                # save_path = saver.save(sess, '{}{}/model.ckpt'.format(self.model_dir, self.id))
                # saver.save(sess, '{}/model'.format(self.save_path))
                np.save('path/{}_idx'.format(self._dataset), train_mask)
                self.get_emb(sess, train_mask,self.batch_size)
                self.__save_path(sess, train_mask, self.batch_size)
                exit()
                return  0, 0
            else:
                predY = self.eval_accuracy(sess, test_mask, largebatchsize=self.batch_size)
                # correct_pred = np.where((Y_test.argmax(-1) + predY) > 1)[0]
                # self.__save_path(sess, test_mask, 128)
                # np.save("paths/{}_correct".format(self.id), correct_pred)
                # np.save("paths/{}_sel".format(self.id), test_mask)
                # print(Y_test.shape, predY.shape)
                Y_test = Y_test.toarray()

                num_of_label = Y_test.shape[1]
                precision = np.zeros(num_of_label)
                recall = np.zeros(num_of_label)
                F1 = np.zeros(num_of_label)
                for l in range(num_of_label):
                    # prediction integer labels
                    try:
                        precision[l] = precision_score(Y_test[:, l], predY[:, l], average='binary')
                        recall[l] = recall_score(Y_test[:, l], predY[:, l], average='binary')
                        F1[l] = f1_score(Y_test[:, l], predY[:, l], average='binary')
                    except:
                        precision[l] = precision_score(Y_test[:, l], predY[:, l].toarray(), average='binary')
                        recall[l] = recall_score(Y_test[:, l], predY[:, l].toarray(), average='binary')
                        F1[l] = f1_score(Y_test[:, l], predY[:, l].toarray(), average='binary')

                acc = accuracy_score(Y_test, predY)
                print(acc)
                try:
                    score_micro = [precision_score(Y_test, predY, average='micro'),
                                   recall_score(Y_test, predY, average='micro'),
                                   f1_score(Y_test, predY, average='micro')]
                    score_macro = [precision_score(Y_test, predY, average='macro'),
                                   recall_score(Y_test, predY, average='macro'),
                                   f1_score(Y_test, predY, average='macro')]
                except:
                    score_micro = [precision_score(Y_test, predY.toarray(), average='micro'),
                                   recall_score(Y_test, predY.toarray(), average='micro'),
                                   f1_score(Y_test, predY.toarray(), average='micro')]
                    score_macro = [precision_score(Y_test, predY.toarray(), average='macro'),
                                   recall_score(Y_test, predY.toarray(), average='macro'),
                                   f1_score(Y_test, predY.toarray(), average='macro')]


                # return accuracy_score(Y_test, predY), f1_score(Y_test, predY, average='macro')  # , accuracy_score(Y_test.argmax(-1), predY_lg))
                return acc, score_micro, score_macro, [precision, recall, F1]

            # if verbose: print("Model saved in file: %s" % save_path)




            # classifier = LogisticRegression(max_iter=1000)

            # X_train = self.get_emb(sess, train_mask, largebatchsize=500)
            # X_test = self.get_emb(sess, test_mask, largebatchsize=500)
            # classifier.fit(X_train, Y_train.argmax(-1))
            # predY_lg = classifier.predict(X_test)



        # get edge attenstion weights . called per epoch

    def __predict(self, trueX):
        '''
        measure the semi-supervised accuracy

        trueX -- input 2D tensor (features)

        return a vector tensor
        '''
        return self.__classify(trueX)

    def get_emb(self, sess, X, largebatchsize=200):
        '''
        X     -- 2D feature matrix (n_samples x DX)
        Y     -- 2D one-hot matrix (n_samples x DY) or 1D labels (n_samples)
        onehot -- True: Y is one-hot
                  False: Y is integer labels

        test the classsfication accuracy
        '''
        self.is_train = False
        embX = []  # np.zeros(Y.shape)

        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))
        # saver = tf.train.import_meta_graph("model.meta")
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        # saver = tf.train.Saver()
        # saver.restore(sess, "{}/model".format(self.save_path))


        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            # with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            #     pred_Y = tf.get_variable("predY_{}".format(len(batch)), shape=[len(batch), self.config['dim_y']], initializer=tf.ones_initializer)
            #     sess.run(pred_Y.initializer)
            embX.append(sess.run(self.get_embd,
                                 feed_dict={self.X: X[batch],
                                            self.is_training: False,
                                            self.get_path: False}))

        print(embX[0].shape, embX[1].shape)
        emb = np.concatenate(embX,-2)
        np.save("path/{}_emb".format(self._dataset), emb)
        # try:
        #     shutil.rmtree(self.save_path)
        # except OSError as e:
        #     print ("Error: %s - %s." % (e.filename, e.strerror))
        # return emb


    def __save_path(self, sess, X, largebatchsize=200):
        '''
        X     -- 2D feature matrix (n_samples x DX)
        Y     -- 2D one-hot matrix (n_samples x DY) or 1D labels (n_samples)
        onehot -- True: Y is one-hot
                  False: Y is integer labels

        test the classsfication accuracy
        '''
        self.is_train = False
        pathX_node = []  # np.zeros(Y.shape)
        pathX_edge = []
        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))
        # saver = tf.train.import_meta_graph("model.meta")
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        # saver = tf.train.import_meta_graph('{}/model.meta'.format(self.save_path))
        # pdb.set_trace()
        # saver.restore(sess, "{}/model".format(self.save_path))



        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            # with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            #     pred_Y = tf.get_variable("predY_{}".format(len(batch)), shape=[len(batch), self.config['dim_y']], initializer=tf.ones_initializer)
            #     sess.run(pred_Y.initializer)
            paths = sess.run(self.get_pt,
                             feed_dict={self.X: X[batch],
                                        self.is_training: False,
                                        self.get_path: True})

            pathX_node.append(paths)

        # pdb.set_trace()
        path_node = np.vstack(pathX_node)
        np.save("path/{}_node".format(self._dataset), path_node)
        # try:
        #     shutil.rmtree(self.save_path)
        # except OSError as e:
        #     print ("Error: %s - %s." % (e.filename, e.strerror))



    def eval_accuracy(self, sess, X, largebatchsize=500):
        '''
        X     -- 2D feature matrix (n_samples x DX)
        Y     -- 2D one-hot matrix (n_samples x DY) or 1D labels (n_samples)
        onehot -- True: Y is one-hot
                  False: Y is integer labels

        test the classsfication accuracy
        '''
        self.is_train = False
        predY = []  # np.zeros(Y.shape)

        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))
        # self.config['A'] = 10
        # saver = tf.train.import_meta_graph("model.meta")
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        # saver = tf.train.import_meta_graph('{}/model.meta'.format(self.save_path))
        # pdb.set_trace()
        # saver.restore(sess, "{}/model".format(self.save_path))
        # pdb.set_trace()
        # pdb.set_trace()

        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            # with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            #     pred_Y = tf.get_variable("predY_{}".format(len(batch)), shape=[len(batch), self.config['dim_y']], initializer=tf.ones_initializer)
            #     sess.run(pred_Y.initializer)
            predY.append(sess.run(self.pred,
                                  feed_dict={self.X: X[batch],
                                             self.is_training: False,
                                             self.get_path: False}))
        predY = np.vstack(predY)

        ## Try to remove tree; if failed show an error using try...except on screen
        # try:
        #     shutil.rmtree(self.save_path)
        # except OSError as e:
        #     print ("Error: %s - %s." % (e.filename, e.strerror))
        return predY


# ref_features = load_npz("../final_extraction/{}/{}_matrices/lsi_c.npz".format(_dataset,_dataset)).toarray()
# model = edge_att(ref_features.shape[0])# model = edge_att(ref_features.shape[0])