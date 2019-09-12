
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, time
os.environ['PYTHONHASHSEED'] = '2018'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

glo_seed = 2018

import random as rn
import tensorflow as tf
import numpy as np
from scipy.sparse import hstack, csr_matrix, vstack
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.sparse import load_npz, save_npz
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_distances



class MLGWalk():
    '''
    Multi-label Graph Walk
    '''

    def __init__(self, FLAGS, num_of_labels):
        '''
        Initializer
        FLAGS -- input parameters
        num_of_labels -- total number of labels
        '''

        # For reproducibility
        rn.seed(glo_seed)
        self.rng = np.random.RandomState(seed=glo_seed)
        np.random.seed(glo_seed)
        tf.set_random_seed(glo_seed)


        #Initialize parameters
        self.dtype = tf.float32
        self.FLAGS = FLAGS

        # the hyper-parameters
        self.config = OrderedDict()
        self.config['l_dim'] = FLAGS.l_dim
        self.config['walk_len'] = FLAGS.walk_len
        self.config['beta'] = FLAGS.beta
        self.config['lrate'] = FLAGS.lrate
        self.config['max_neighbors'] = FLAGS.max_neighbors
        self.config['gamma'] = FLAGS.gamma
        self.config['alpha'] = FLAGS.alpha

        #other model parameters 
        self.config['num_walks'] = FLAGS.num_walks
        self.config['dim_y'] = num_of_labels
        self.config['transductive'] = FLAGS.transductive
        self.config['activation'] = tf.nn.relu
        
        # place holders
        self.X = tf.placeholder(tf.int64, shape=(None,))
        self.is_training = tf.placeholder(tf.bool)
        self.get_path = tf.placeholder(tf.bool)
        self.gamma = tf.constant(self.config['gamma'])

        self.T = tf.constant(float(self.config['walk_len']))
        self.range = tf.Variable(tf.range(0, self.config['walk_len'], 1, dtype=self.dtype), trainable=False)

        # identifier
        self.id = ('{0}_{1}_{2}_{3}_{4}'.format(
            self.FLAGS.dataset,
            self.config['walk_len'],
            self.config['l_dim'],
            self.config['max_neighbors'],
            time.time()))

    
        self.is_train = True


    #load the node and edge attributes 
    def load_data(self, test_nodes):

        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            node_features = load_npz("{}/{}_matrices/lsi.npz".format(self.FLAGS.dataset_dir,self.FLAGS.dataset)).toarray()
            node_features = Normalizer().fit_transform(node_features)
            node_features = np.vstack((np.zeros(node_features.shape[-1]), node_features))

            self.node_emb = tf.get_variable(name="node_emb", shape=node_features.shape, initializer=tf.constant_initializer(node_features), dtype=self.dtype,
                                                trainable=False, use_resource=True)
            self.config['dim_Av'] = self.node_emb.get_shape().as_list()[-1]

            if self.FLAGS.has_edge_attr:
                try:
                    edge_features = load_npz("{}/{}_matrices/lsi_context.npz".format(self.FLAGS.dataset_dir,self.FLAGS.dataset)).toarray()
                    edge_features = np.vstack((np.zeros(edge_features.shape[-1]), edge_features))
                    self.edge_features = Normalizer().fit_transform(edge_features)
                    self.edge_emb_init= tf.placeholder(tf.float32, shape=self.edge_features.shape)
                    self.edge_emb = tf.Variable(self.edge_emb_init, name="edge_emb", trainable=False, use_resource=True)
                    self.has_edge_attr = True

                except:
                    print("Error reading the reference file. Defaulting to text only")
                    self.edge_emb = None
                    self.has_edge_attr = False
            else:
                self.edge_emb = None
                self.has_edge_attr = False


            self.Y = tf.placeholder(self.dtype, shape=(None, self.config['dim_y']))
            self.setup_lookup(test_nodes) #setup 
            self.attention_node = tf.Variable(tf.ones([self.config['dim_y'],self.config['num_walks'], self.FLAGS.batchsize]),validate_shape=False) #tf.get_variable("att_vect", shape=[self.config['num_walks'], 128])
            self.attention_edge = tf.Variable(tf.ones([self.config['dim_y'], self.config['num_walks'], self.FLAGS.batchsize]),validate_shape=False) #tf.get_variable("att_vect", shape=[self.config['num_walks'], 128])

            self.cost = self.__cost(self.X)
            self.get_embd = self.get_embbeding(self.X)
            self.get_pt = self.__get_walk(self.X)
            self.pred = self.__predict(self.X)


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


    def setup_lookup(self, test_nodes):
        '''
        Setup the lookup tables required for smooth run
        '''

        edgelist = open("{}/{}.edgelist".format(self.FLAGS.dataset_dir,self.FLAGS.dataset), "rU").read().split("\n")[:-1]
        neighbor = {}
        edge_tensor = [0,0]
        iter = 2
        neighbor[0] = [[], [[0, 1]]]
        neighbor[0] = [[[0, 1]], []]
        for edge in edgelist:
            edgei = edge.split('\t')
            s, t = map(int, edgei[:2])

            s, t = s+1, t+1
            if neighbor.has_key(t):
                neighbor[t][1].append([iter])
            else:
                neighbor[t] = [[],[[iter]]]

            iter += 1
            if neighbor.has_key(s):
                neighbor[s][0].append([iter])
            else:
                neighbor[s] = [[[iter]], []]


            edge_tensor.extend((s,t))
            iter += 1

        edges_per_node = np.zeros((len(neighbor), self.config['max_neighbors']))
        for key, value in neighbor.iteritems():
            value[0] = np.array(value[0])
            value[1] = np.array(value[1])
            half = int(self.config['max_neighbors'] / 2)
            if value[0].shape[0] > 0:
                if value[0].shape[0] <= half:
                    edges_per_node[key, :value[0].shape[0]] = value[0][:, 0]
                    space = self.config['max_neighbors'] - value[0].shape[0]
                    if value[1].shape[0] > 0:
                        others = value[1][:, 0]
                        if others.shape[0] >= space:
                            others_samp = rn.sample(others, space)
                            edges_per_node[key, value[0].shape[0]:value[0].shape[0] + space] = others_samp
                        else:
                            edges_per_node[key, value[0].shape[0]:value[0].shape[0] + others.shape[0]] = others
                else:
                    rank = value[0][:, 0]
                    samp = rn.sample(rank, half)
                    cur = list(set(rank).difference(samp))
                    edges_per_node[key, :half] = samp
                    if value[1].shape[0] < 1:
                        edges_per_node[key, half:half+len(cur)] = cur[:half]
                    else:
                        others = value[1][:, 0]
                        if others.shape[0] >= half:  
                            others_samp = rn.sample(others, half)
                            edges_per_node[key, half:] = others_samp
                        else:
                            edges_per_node[key, half:(half + others.shape[0])] = others
                            space = self.config['max_neighbors'] - (half + others.shape[0])
                            space = len(cur) if len(cur) < space else space
                            edges_per_node[key, (half + others.shape[0]):(half + others.shape[0])+space] = cur[:space]
                    
            elif value[1].shape[0] > 0:
                others = value[1][:, 0]
                if others.shape[0] >= self.config['max_neighbors']:
                    others_samp = rn.sample(others, self.config['max_neighbors'])
                    edges_per_node[key, :] = others_samp
                else:
                    edges_per_node[key, :others.shape[0]] = others
        # with tf.device('/cpu:0'):
        if self.config['transductive']:
            test_mask = edges_per_node > 0
        else:
            test_mask = ~np.isin(np.array(edge_tensor)[edges_per_node.astype(np.int32)], test_nodes)
            null_neighboors = edges_per_node > 0
            test_mask = np.logical_and(test_mask, null_neighboors)
        self.edges_per_node = tf.convert_to_tensor(edges_per_node, tf.int64)#self.denseNDArrayToSparseTensor(edges_per_node)
        self.edge_tensor = tf.convert_to_tensor(edge_tensor, tf.int64)
        self.test_mask = tf.convert_to_tensor(test_mask, tf.bool)


    def gather_cols3D(self, params, indices):
        '''
            Select specific columns per row for 3D matrix
        '''
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1], [-1]) + indices, [-1])
        partitions = tf.reduce_sum(tf.one_hot(i_flat, tf.shape(p_flat)[0], dtype='int32'), 0)
        col_sel = tf.dynamic_partition(p_flat, partitions, 2)
        col_sel = col_sel[1]
        return tf.reshape(col_sel,[p_shape[0], 1])


    def gather_cols4D(self, params, indices):
        '''
            Select specific columns per row for 4D matrix
        '''
        p_shape = tf.shape(params)
        A = params.get_shape().as_list()[:2]
        p_flat = tf.reshape(params, [-1])
        indices = tf.reshape(indices, [-1])
        i_flat = tf.reshape(tf.range(0, p_shape[1]*p_shape[0]*p_shape[2]) * p_shape[3], [-1]) + indices
        partitions = tf.reduce_sum(tf.one_hot(i_flat, tf.shape(p_flat)[0], dtype='int32'), 0)
        col_sel = tf.dynamic_partition(p_flat, partitions, 2)
        col_sel = col_sel[1]
        return tf.reshape(col_sel,[A[0],A[1], p_shape[2], 1])


    def dense(self, inputs, output_dim, name, is_private=True):

        '''
            Dense layer

            inputs -- input vectors
            output_dim -- output dimension 
            name -- name of the operation
            is_private -- independent weights per label agent
        '''
        shape = inputs.get_shape().as_list()
        if len(shape) > 4:
            if is_private:
                W = tf.get_variable(name='W_{}'.format(name) ,
                                    initializer = lambda: tf.glorot_uniform_initializer()(( shape[0], shape[-1], output_dim)))
                b = tf.get_variable(name='b_{}'.format(name) , initializer = lambda: -1*tf.ones_initializer()(output_dim))

                return tf.nn.bias_add(tf.einsum('abilj,ajk->abilk', inputs, W), b)
            else:
                W = tf.get_variable(name='W_{}'.format(name) ,
                                    initializer = lambda: tf.glorot_uniform_initializer()((shape[-1], output_dim)))
                b = tf.get_variable(name='b_{}'.format(name) , initializer = lambda: -1*tf.ones_initializer()(output_dim))

                return tf.nn.bias_add(tf.einsum('abilj,jk->abilk', inputs, W), b)

        else:
            W = tf.get_variable(name='W_{}'.format(name) ,
                                initializer = lambda: tf.glorot_uniform_initializer()((shape[0],  shape[-1], output_dim)))
            b = tf.get_variable(name='b_{}'.format(name) , initializer = lambda: -1*tf.ones_initializer()(output_dim))

            return tf.nn.bias_add(tf.einsum('abij,ajk->abik', inputs, W), b)


    def sample_neighbor_walk(self, current_x, current_emb, h, t, reuse=tf.AUTO_REUSE):
        '''
            current_x -- current nodes (v^t)
            current_emb -- current node feature embedding (x^t)
            h -- history context
            t -- time step

            Sample the next nodes to visit (Step procedure)
        '''

        #Get node neighbors
        neighbors = tf.gather(self.edges_per_node, current_x)
        #Get mask which removing test nodes and dummy node neighbors from node neighborhood
        mask_neighbors = tf.gather(self.test_mask, current_x)
        #If training phase: mask/remove only dummy nodes, else remove dummy and test nodes (if inductive)
        mask = tf.cond(self.is_training, lambda : mask_neighbors, lambda : tf.greater(neighbors,0 ))

        #Setup inputs to the score network
        h = tf.tile(tf.expand_dims(h, 3), [1,1, 1, self.config['max_neighbors'], 1])
        neighbor_node_emb = tf.nn.embedding_lookup(self.node_emb, tf.gather(self.edge_tensor, neighbors))
        current_emb = tf.tile(tf.expand_dims(current_emb, 3), [1,1,1, self.config['max_neighbors'], 1])
        if self.has_edge_attr:
            neighbor_edge_emb = tf.nn.embedding_lookup(self.edge_emb,tf.div(neighbors,2))
            neighbor_emb_act = tf.add_n((neighbor_edge_emb, current_emb, neighbor_node_emb))
        else:
            neighbor_emb_act = tf.add_n((current_emb, neighbor_node_emb))
        att_emb = tf.concat((h, neighbor_emb_act), -1)

        #Score network
        neighbors_weight = tf.squeeze(tf.keras.backend.hard_sigmoid(self.dense(att_emb, 1, name='e_e_dense2')),-1)
        #Zero out the masked neighbors and neighbors with score < 0.5
        neighbors_weight = tf.multiply(neighbors_weight,tf.cast(mask, tf.float32))
        filter_neighbors = tf.greater_equal(neighbors_weight, 0.5)
        mask2 = tf.logical_and(filter_neighbors, mask)
        #Sample from the probability distribution
        # neighbors_weight = tf.div_no_nan(neighbors_weight , tf.reduce_sum(neighbors_weight))
        neighbors_weight = tf.div_no_nan(neighbors_weight , tf.reduce_sum(neighbors_weight, -1, keepdims=True))

        if self.FLAGS.variant == "mlgw_i":
            next_id_sample = tf.expand_dims(tf.distributions.Categorical(probs=neighbors_weight).sample(), -1)
        else:
            private_policy = tf.distributions.Categorical(logits=tf.log(tf.clip_by_value(neighbors_weight,1e-10,1.0)))
            #Global policy
            att_emb_glob = self.dense(att_emb, self.config['l_dim'], name='e_e_dense2_emb', is_private= True)
            neighbors_weight_glob = tf.squeeze(tf.keras.backend.hard_sigmoid(self.dense(att_emb_glob, 1, name='e_e_dense2_public', is_private=False )),-1)
            neighbors_weight_glob = tf.multiply(neighbors_weight_glob,tf.cast(mask, tf.float32))
            neighbors_weight_glob = tf.div_no_nan(neighbors_weight_glob , tf.reduce_sum(neighbors_weight_glob, -1, keepdims=True))
            global_policy = tf.distributions.Categorical(logits=tf.log(tf.clip_by_value(neighbors_weight_glob,1e-10,1.0)))

            if self.FLAGS.variant == "mlgw_kl":
                next_id_sample = tf.expand_dims(private_policy.sample(), -1)
            elif self.FLAGS.variant == "mlgw_kl+":
                final_weight = tf.cond(self.is_training, lambda : tf.multiply(neighbors_weight, neighbors_weight_glob), lambda : neighbors_weight)
                joint_policy = tf.distributions.Categorical(logits=tf.log(tf.clip_by_value(final_weight,1e-10,1.0)))
                next_id_sample = tf.expand_dims(joint_policy.sample(), -1)
            else:
                print("Unknown variant option: {}".format(self.FLAGS.variant))
                exit()

            

        #Obtain the sampled next node to visit
        next_id = tf.batch_gather(neighbors, next_id_sample)
        next_id = tf.nn.embedding_lookup(self.edge_tensor, next_id)

        #Aggregate the embeddings of the neighbors with score > 0.5 (C_n^t)
        neighbor_emb = tf.reduce_sum(tf.multiply(neighbor_node_emb, tf.expand_dims(tf.cast(mask2, tf.float32), -1)),3)
        #Just to further prevent sampling a masked node in the unlikely event that all had zero probabilities
        is_sample_masked = tf.batch_gather(mask, next_id_sample)
        non_isolated_nodes = tf.logical_and(tf.reduce_any(mask, -1), tf.squeeze(is_sample_masked,-1))
        next_id = tf.add(tf.multiply(tf.squeeze(next_id,-1),tf.cast(non_isolated_nodes, tf.int64)) , tf.multiply(current_x,tf.cast(~non_isolated_nodes, tf.int64)))

        if self.FLAGS.variant == "mlgw_kl+":
            likelihood = tf.squeeze(self.gather_cols4D(final_weight, next_id_sample),[-1])
        else:
            likelihood = tf.squeeze(self.gather_cols4D(neighbors_weight, next_id_sample),[-1])
        likelihood = tf.multiply(tf.pow(self.gamma, (self.T - t)), tf.log(tf.clip_by_value(likelihood,1e-10,1.0)))

        if self.FLAGS.variant == "mlgw_i":
            return tf.expand_dims(next_id,-1), neighbor_emb, tf.expand_dims(likelihood,-1), None
        elif "mlgw_kl" in self.FLAGS.variant:
            # entropy = -self.config['beta'] * tf.reduce_mean(tf.losses.log_loss(labels=neighbors_weight, predictions=neighbors_weight_glob, reduction=tf.losses.Reduction.NONE), -1)
            KL = tf.multiply(tf.pow(self.gamma, (self.T - t)), tf.distributions.kl_divergence(private_policy, global_policy))
            return tf.expand_dims(next_id,-1), neighbor_emb, tf.expand_dims(likelihood,-1),  tf.expand_dims(KL,-1)


    def GRU(self, trueX):

        def forward(input, t):
            """Perform a forward pass.

            Arguments
            ---------
            h_tm1: np.matrix
                The hidden state at the previous timestep (h_{t-1}).
            x_t: np.matrix
                The input vector.
            c_t: np.matrix
                The aggregated neighborhood vector.
            """

            h_tm1 = input[:,:,:,:self.config['l_dim']]
            x = tf.cast(input[:,:,:,self.config['l_dim']], tf.int64)
            h_tm1 = tf.cond(self.is_training, lambda : h_tm1 * self.dropout_recurrent, lambda : h_tm1)


            x_t = tf.nn.embedding_lookup(self.node_emb, x)
            next_x, c_t, likelihood, KL = self.sample_neighbor_walk(x, x_t, h_tm1, t)

            x_t = tf.concat([x_t, c_t], -1)
            
            zr_t = tf.keras.backend.hard_sigmoid(self.dense(tf.concat([x_t, h_tm1],-1), self.config['l_dim']*2, name='zr'))
            z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)
            r_state = r_t * h_tm1
            h_proposal = tf.tanh(self.dense(tf.concat([x_t, r_state],-1), self.config['l_dim'], name='h'))


            # Compute the next hidden state
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

            if self.FLAGS.variant == "mlgw_i":
                return tf.concat([h_t, tf.cast(next_x, self.dtype), likelihood, x_t],-1)
            elif "mlgw_kl" in self.FLAGS.variant:
                return tf.concat([h_t, tf.cast(next_x, self.dtype), likelihood, KL, x_t],-1)
            else:
                print("Unknown variant option: {}".format(self.FLAGS.variant))
                exit()

        # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
        dummy_emb = tf.tile(tf.expand_dims(tf.cast(trueX, self.dtype),-1), [1,1,1,self.config['dim_Av']])
        shape = dummy_emb.get_shape().as_list()
        h_0 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(shape[0],shape[1], self.config['dim_Av'], self.config['l_dim'])),
                        name='h_0' )
        next_x0 = tf.expand_dims(tf.cast(trueX, self.dtype),-1)

        if self.FLAGS.variant == "mlgw_i":
            concat_tensor = tf.concat([h_0, next_x0, next_x0, dummy_emb, dummy_emb], -1)
        elif "mlgw_kl" in self.FLAGS.variant:
            concat_tensor = tf.concat([h_0, next_x0, next_x0, next_x0,  dummy_emb, dummy_emb], -1)
        else:
            print("Unknown variant option: {}".format(self.FLAGS.variant))
            exit()
        

        if self.is_train == True:
            dropout_hidden = tf.nn.dropout(h_0[0,0,:,:], 0.80, name='dropout2')
        self.dropout_recurrent = tf.get_default_graph().get_tensor_by_name('{}/Floor:0'.format("/".join(dropout_hidden.name.split("/")[:-1])))

        h_t = tf.scan(forward, self.range, initializer = concat_tensor,parallel_iterations=20,
                      name='h_t_transposed' )

        if self.FLAGS.variant == "mlgw_i":
            h_t_b = self.BGRU(tf.reverse(h_t[:,:,:,:,self.config['l_dim'] +2:], [0]))
        elif "mlgw_kl" in self.FLAGS.variant:
            h_t_b = self.BGRU(tf.reverse(h_t[:,:,:,:,self.config['l_dim'] +3:], [0]))
        else:
            print("Unknown variant option: {}".format(self.FLAGS.variant))
            exit()

        
        ht =  h_t[-1,:,:,:,:self.config['l_dim']] + h_t_b
        output = tf.cond(self.get_path, lambda : tf.transpose(h_t[:,:,:, :,self.config['l_dim']], perm=[3,1,2,0]), lambda : ht)
        
        if self.FLAGS.variant == "mlgw_i":
            return output, tf.reduce_sum(h_t[:-1,:, :,:,self.config['l_dim']+1],0), None
        elif "mlgw_kl" in self.FLAGS.variant:
            return output, tf.reduce_sum(h_t[:-1,:, :,:,self.config['l_dim']+1],0), tf.reduce_sum(h_t[:-1,:, :,:,self.config['l_dim']+2],0)
        else:
            print("Unknown variant option: {}".format(self.FLAGS.variant))
            exit()



    def BGRU(self, node_emb):


        def backward(h_tm1, x_t):
            """Perform a backward pass.

            Arguments
            ---------
            h_tm1: np.matrix
                The hidden state at the previous timestep (h_{t-1}).
            x_t: np.matrix
                The concatenation of the input and corresponding neighborhood vectors.
            """


            h_tm1 = tf.cond(self.is_training, lambda: h_tm1 * self.dropout_recurrent_b, lambda: h_tm1)
            zr_t = tf.keras.backend.hard_sigmoid(self.dense(tf.concat([x_t, h_tm1],-1), self.config['l_dim']*2, name ='zr_b'))
            z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)
            r_state = r_t * h_tm1
            h_proposal = tf.tanh(self.dense(tf.concat([x_t, r_state],-1), self.config['l_dim'], name='h_b'))
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

            return h_t



        # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
        shape = node_emb.get_shape().as_list()
        h_0_b = tf.matmul(node_emb[0, :, :, :, :], tf.zeros(dtype=tf.float32, shape=(shape[1],shape[2], self.config['dim_Av']*2, self.config['l_dim'])),
                          name='h_0_b' )
        if self.is_train == True:
            dropout_hidden = tf.nn.dropout(h_0_b[0,:,:], 0.80, name='dropout2b' )
            self.dropout_recurrent_b = tf.get_default_graph().get_tensor_by_name('{}/Floor:0'.format("/".join(dropout_hidden.name.split("/")[:-1])))

        h_t_transposed_b = tf.scan(backward, node_emb, initializer = h_0_b,parallel_iterations=20, name='h_t_transposed_b' )
        return h_t_transposed_b[-1,:,:,:,:]


    def __cost(self, trueX):
        '''
        compute the cost tensor

        trueX -- input X (1D tensor)
        trueY -- input Y (2D tensor)

        return 1D tensor of cost (batch_size)
        '''

        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):


            X = tf.expand_dims(tf.expand_dims(trueX, 0),0 )
            X = tf.tile(X, [self.config['dim_y'], self.config['num_walks'],1])

            Y = tf.transpose(tf.expand_dims(self.Y,-1), perm=[1,2,0])
            Y = tf.tile(Y, [1,self.config['num_walks'], 1])

            # X -> Z
            Z, likelihood, KL = self.GRU(X)
            Z = tf.reshape(Z, [self.config['dim_y'], self.config['num_walks'], -1,self.config['l_dim']])
            
            log_pred_Y = tf.squeeze(self.dense(Z, 1, name='Z2y'),-1)

            reward = tf.cast(tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(log_pred_Y), 0.5), self.dtype), Y), tf.float32)
            reward = 2*(reward-0.5)

            _cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(Y, self.dtype), logits=log_pred_Y)

            
            if self.FLAGS.variant == "mlgw_i":
                M_loss = tf.reduce_mean(_cost - reward * (-self.config['beta'] * likelihood), 1)
            elif "mlgw_kl" in self.FLAGS.variant:
                M_loss = tf.reduce_mean(_cost - reward * ((-self.config['beta']  *likelihood) - (self.config['alpha'] * KL)), 1)
            else:
                print("Unknown variant option: {}".format(self.FLAGS.variant))
                exit()

            return tf.reduce_mean(tf.reduce_mean(M_loss, 0))

    def __classify(self, trueX, reuse=tf.AUTO_REUSE):
        '''
        classify input 1D tensor

        return 2D tensor (integer class labels)
        '''
        with tf.variable_scope("cost", reuse=True):


            # X -> Z
            X = tf.expand_dims(tf.expand_dims(trueX, 0),0)
            X = tf.tile(X, [self.config['dim_y'], self.config['num_walks'],1])
            Z, _, _ = self.GRU(X)
            Z = tf.reshape(Z, [self.config['dim_y'], self.config['num_walks'], -1,self.config['l_dim']])

            log_pred_Y = tf.squeeze(self.dense(Z, 1, name='Z2y'),-1)
            Y = tf.reduce_mean(tf.nn.sigmoid(log_pred_Y),1)
            Y = tf.cast(tf.greater_equal(Y, 0.5), tf.int32)

            return tf.transpose(Y)


    def get_embbeding(self, trueX, reuse=tf.AUTO_REUSE):
        '''
        Return latent vectors for nodes
        '''
        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):

            # X -> Z
            X = tf.expand_dims(tf.expand_dims(trueX, 0),0)
            X = tf.tile(X, [self.config['dim_y'], self.config['num_walks'],1])
            Z, _, _ = self.GRU(X)
            Z = tf.reshape(Z, [self.config['dim_y'], self.config['num_walks'], -1,self.config['l_dim']])

            return Z


    def __get_walk(self, trueX, reuse=tf.AUTO_REUSE):
        '''
        return walk paths taken to classify nodes
        '''
        X = tf.expand_dims(tf.expand_dims(trueX, 0),0)
        X = tf.tile(X, [self.config['dim_y'], self.config['num_walks'],1])
        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            walk, _, _ = self.GRU(X)
            walk = tf.reshape(walk, [-1,self.config['dim_y'],self.config['num_walks'], self.config['walk_len']])
            return walk


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


    def train(self, labels, samples):
        '''
            Training and evaluation 
            labels -- Target node labels
            samples -- Training and testing nodes 
        '''

        train_mask, test_mask = samples
        if self.FLAGS.save:
            if not os.path.exists('paths'):
                os.mkdir('paths')
            train_mask = np.hstack((train_mask, test_mask))
            save_npz('paths/{}_idx'.format(self.id), csr_matrix(train_mask))
            if verbose: print("Path indecies saved in file: %s" % "path/{}".format(self.FLAGS.dataset))
            allidx = np.arange(train_mask.shape[0])
            Y_train = labels[train_mask, :]
            nbatches = int(np.ceil(train_mask.shape[0] / self.FLAGS.batchsize))
        else:
            allidx = np.arange(train_mask.shape[0])
            Y_train = labels[train_mask, :]
            Y_test = labels[test_mask, :]
            nbatches = int(np.ceil(train_mask.shape[0] / self.FLAGS.batchsize))

        self.is_train = True

        with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
            train_op = tf.train.AdamOptimizer(self.config['lrate']).minimize(self.cost)

        #TF config setup
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        glob_init = tf.global_variables_initializer()

        with tf.Session(config=config) as sess:

            #initialize variables
            if self.has_edge_attr:
                sess.run(glob_init, feed_dict={self.edge_emb_init: self.edge_features})
            else:
                sess.run(glob_init)
            sess.run(tf.local_variables_initializer())
            sess.graph.finalize()  # Graph is read-only after this statement.


            for e in range(self.FLAGS.epochs):
                self.rng.shuffle(allidx)
                epoch_cost = 0

                for batchl in self.fetch_batches(allidx, nbatches, self.FLAGS.batchsize):

                    if self.FLAGS.verbose:
                        epoch_cost += sess.run([train_op, self.cost],
                                               feed_dict={self.X: train_mask[batchl],
                                                          self.Y: Y_train[batchl, :].toarray(),
                                                          self.is_training: False,
                                                          self.get_path: False})[1]
                    else:
                        sess.run(train_op,
                                 feed_dict={self.X: train_mask[batchl],
                                            self.Y: Y_train[batchl, :].toarray(),
                                            self.is_training: False,
                                            self.get_path: False})
                epoch_cost /= nbatches

                if self.FLAGS.verbose:
                    cost_only = True
                    if cost_only:
                        print('[{0:5d}] E={1:.8f}\n'.format(e, epoch_cost))
                    else:
                        train_acc = sess.run(self.acc,
                                             feed_dict={self.X: train_mask[0:self.FLAGS.batchsize],
                                                        self.Y: Y_train[0:self.FLAGS.batchsize, :].toarray(),
                                                        self.is_training: False,
                                                        self.get_path: False})

                        print( '[{0:5d}]  TrainE={1:.4f}  TrainAcc={2:.4f}'.format( e, epoch_cost, train_acc ))

            if self.FLAGS.save:
                np.save('path/{}_idx'.format(self.FLAGS.dataset), train_mask)
                self.get_emb(sess, train_mask,self.FLAGS.batchsize)
                self.__save_path(sess, train_mask, self.FLAGS.batchsize)
                if verbose: print("Path saved in file: %s" % "path/{}_node".format(self.FLAGS.dataset))
                exit()
            else:
                predY = self.prediction(sess, test_mask, largebatchsize=self.FLAGS.batchsize)
                Y_test = Y_test.toarray()

                precision = np.zeros(self.config['dim_y'])
                recall = np.zeros(self.config['dim_y'])
                F1 = np.zeros(self.config['dim_y'])

                #Evaluation of inidividual label prediction performance
                for l in range(self.config['dim_y']):
                    try:
                        precision[l] = precision_score(Y_test[:, l], predY[:, l], average='binary')
                        recall[l] = recall_score(Y_test[:, l], predY[:, l], average='binary')
                        F1[l] = f1_score(Y_test[:, l], predY[:, l], average='binary')
                    except:
                        precision[l] = precision_score(Y_test[:, l], predY[:, l].toarray(), average='binary')
                        recall[l] = recall_score(Y_test[:, l], predY[:, l].toarray(), average='binary')
                        F1[l] = f1_score(Y_test[:, l], predY[:, l].toarray(), average='binary')

                #General prediction performance
                acc = accuracy_score(Y_test, predY)
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


                return acc, score_micro, score_macro, [precision, recall, F1]


    def __predict(self, trueX):
        '''
        predict the node labels

        trueX -- input 1D tensor (nodes)

        return a 2D label prediction tensor
        '''
        return self.__classify(trueX)

    def get_emb(self, sess, X, largebatchsize=200):
        '''
        Get learned node embeddings
        X     -- 1D vector (n_sample nodes)
        emb     -- 2D embedding matrix (n_samples x l_dim)

        '''
        self.is_train = False
        embX = [] 

        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))

        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            embX.append(sess.run(self.get_embd,
                                 feed_dict={self.X: X[batch],
                                            self.is_training: False,
                                            self.get_path: False}))

        print(embX[0].shape, embX[1].shape)
        emb = np.concatenate(embX,-2)
        np.save("path/{}_emb".format(self.FLAGS.dataset), emb)
        

    def __save_path(self, sess, X, largebatchsize=200):
        '''
        X     -- 1D vector (n_sample nodes)
        path_node  -- 3D path matrix (n_samples x num-walks x num-length)

        Save learned walk path
        '''
        self.is_train = False
        pathX_node = []  # np.zeros(Y.shape)
        pathX_edge = []
        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))

        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            paths = sess.run(self.get_pt,
                             feed_dict={self.X: X[batch],
                                        self.is_training: False,
                                        self.get_path: True})

            pathX_node.append(paths)

        path_node = np.vstack(pathX_node)
        np.save("path/{}_node".format(self.FLAGS.dataset), path_node)


    def prediction(self, sess, X, largebatchsize=500):
        '''
        X     -- 1D vector (n_sample nodes)
        Y     -- 2D one-hot matrix (n_samples x DY) or 1D labels (n_samples)

        predict node labels
        '''
        self.is_train = False
        predY = [] 

        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))

        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            predY.append(sess.run(self.pred,
                                  feed_dict={self.X: X[batch],
                                             self.is_training: False,
                                             self.get_path: False}))
        predY = np.vstack(predY)

        return predY