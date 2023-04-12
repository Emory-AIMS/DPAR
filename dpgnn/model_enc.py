import numpy as np
import tensorflow.compat.v1 as tf
from . import tf_utils
from . import utils
from .privacy_utils import dp_optimizer
from dpgnn.utils import SparseRowIndexer
import scipy.sparse as sp
import sys

class DPGNN:
    def __init__(self, d, nc, hidden_size, nlayers, nEncLayers, nEncOut, lr,
                 weight_decay, dropout, nEncDropout, clip_bound, sigma, microbatches,
                 dp_sgd=False, enc_dp_sgd=False, sparse_features=True):
        self.nc = nc
        self.sparse_features = sparse_features

        if sparse_features:
            self.batch_feats = tf.sparse_placeholder(tf.float32, None, 'features')
        else:
            self.batch_feats = tf.placeholder(tf.float32, [None, d], 'features')
        self.batch_pprw = tf.placeholder(tf.float32, [None], 'ppr_weights')
        self.batch_idx = tf.placeholder(tf.int32, [None], 'idx')
        self.batch_labels = tf.placeholder(tf.int32, [None], 'labels')

        feats_drop = tf_utils.mixed_dropout(self.batch_feats, dropout)

        #Encoder
        Es = [tf.get_variable('E1', [d, hidden_size])]
        for i in range(nEncLayers - 2):
            Es.append(tf.get_variable(f'E{i+2}', [hidden_size, hidden_size]))
        Eo = tf.get_variable(f'E{nEncLayers}', [hidden_size, nEncOut])

        if sparse_features:
            _h = tf.sparse.sparse_dense_matmul(feats_drop, Es[0])
        else:
            _h = tf.matmul(feats_drop, Es[0])
        for W in Es[1:]:
            _h = tf.nn.relu(_h)
            _h_drop = tf.nn.dropout(_h, rate=nEncDropout)
            _h = tf.matmul(_h_drop, W)

        _h = tf.nn.relu(_h)
        _h = tf.nn.dropout(_h, rate=nEncDropout)
        _h = tf.matmul(_h, Eo)
        self.encoded_feat = _h

        Ex = tf.get_variable('EW', [nEncOut, nc])
        self._encoder_logits = tf.matmul(_h, Ex)
        
        self.encoder_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_labels,
                                                                           logits=self._encoder_logits)
        
        self.encoder_train_preds = tf.argmax(self._encoder_logits, 1)

        self.encoder_variables = [Eo] + Es
        

        # Model
        Ws = [tf.get_variable('W1', [nEncOut, hidden_size])]
        for i in range(nlayers - 2):
            Ws.append(tf.get_variable(f'W{i + 2}', [hidden_size, hidden_size]))
        
        # if sparse_features:
        #     h = tf.sparse.sparse_dense_matmul(self.encoded_feat, Ws[0])
        # else:
        h = tf.matmul(self.encoded_feat, Ws[0])
        for W in Ws[1:]:
            h = tf.nn.relu(h)
            h_drop = tf.nn.dropout(h, rate=dropout)
            h = tf.matmul(h_drop, W)

        self.embedding = h
        Wo = tf.get_variable(f'W{nlayers}', [hidden_size, nc])
        h = tf.nn.relu(self.embedding)
        h_drop = tf.nn.dropout(h, rate=dropout)
        h = tf.matmul(h_drop, Wo)

        self.logits = h

        self.weighted_logits_1 = tf.math.multiply(self.logits, self.batch_pprw[:, None])
        self.weighted_logits = tf.tensor_scatter_nd_add(tf.zeros((tf.shape(self.batch_labels)[0], nc)),
                                                   self.batch_idx[:, None],
                                                   self.logits * self.batch_pprw[:, None])

        self.print = tf.print("orig: ",tf.shape(self.weighted_logits),
                              " mul: ", tf.shape(self.weighted_logits_1), 
                              "pprw", tf.shape(self.batch_pprw), 
                              "idx", self.batch_idx, output_stream=sys.stdout)


        self.preds = tf.argmax(self.weighted_logits, 1)

        loss_per_node = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_labels,
                                                                       logits=self.weighted_logits)

        self.model_variables = [Wo] + Ws

        if dp_sgd:
            noise_multiplier = sigma / clip_bound
            self.loss = loss_per_node
            self.update_op = dp_optimizer.DPAdamGaussianOptimizer(
                l2_norm_clip=clip_bound,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=lr).minimize(self.loss, var_list=self.model_variables)
            
        else:
            l2_reg = tf.add_n([tf.nn.l2_loss(weight) for weight in Ws])
            self.loss = tf.reduce_mean(loss_per_node) + weight_decay * l2_reg
            self.update_op = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=self.model_variables)
            

        if enc_dp_sgd:
            noise_multiplier = sigma / clip_bound
            self.encoder_update_op = dp_optimizer.DPAdamGaussianOptimizer(
                    l2_norm_clip=clip_bound,
                    noise_multiplier=noise_multiplier,
                    num_microbatches=microbatches,
                    learning_rate=lr).minimize(self.encoder_loss)
        else:
            self.encoder_update_op = tf.train.AdamOptimizer(lr).minimize(self.encoder_loss)

        self.cached_train = {}
        self.cached_test = {}

    def feed_for_batch_train(self, attr_matrix, ppr_matrix, labels, key=None):
        if key is None:
            return self.gen_feed(attr_matrix, ppr_matrix, labels)
        else:
            if key in self.cached_train:
                return self.cached_train[key]
            else:
                feed = self.gen_feed(attr_matrix, ppr_matrix, labels)
                self.cached_train[key] = feed
                return feed

    def feed_for_batch_test(self, attr_matrix, ppr_matrix, labels, key=None):
        if key is None:
            return self.gen_feed(attr_matrix, ppr_matrix, labels)
        else:
            if key in self.cached_test:
                return self.cached_test[key]
            else:
                feed = self.gen_feed(attr_matrix, ppr_matrix, labels)
                self.cached_test[key] = feed
                return feed

    def gen_feed(self, attr_matrix, ppr_matrix, labels):
        source_idx, neighbor_idx = ppr_matrix.nonzero()


        batch_attr = attr_matrix[neighbor_idx]
        feed = {
            self.batch_feats: utils.sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
            self.batch_pprw: ppr_matrix[source_idx, neighbor_idx].A1,
            self.batch_labels: labels,
            self.batch_idx: source_idx,
        }
        return feed

    def gen_embed_feed(self, attr_matrix, train_index):
        batch_attr = attr_matrix[train_index]
        feed = {
            self.batch_feats: utils.sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
        }
        return feed

    def _get_logits(self, sess, attr_matrix, nnodes, batch_size_logits=10000):
        logits = []
        for i in range(0, nnodes, batch_size_logits):
            batch_attr = attr_matrix[i:i + batch_size_logits]
            logits.append(sess.run(self.logits,
                                   {self.batch_feats: utils.sparse_feeder(batch_attr) if self.sparse_features else batch_attr}
                                   ))
        logits = np.row_stack(logits)
        return logits

    def feed_for_encoder_batch_train(self, attr_matrix, train_idx, ppr_matrix, labels, key=None):
        if key is None:
            return self.gen_encoder_feed(attr_matrix, train_idx, ppr_matrix, labels)
        else:
            if key in self.cached_train:
                return self.cached_train[key]
            else:
                feed = self.gen_encoder_feed(attr_matrix, train_idx, ppr_matrix, labels)
                self.cached_train[key] = feed
                return feed
    
    def gen_encoder_feed(self, attr_matrix, train_idx, ppr_matrix, labels):
        batch_attr = attr_matrix[train_idx]
        feed = {
            self.batch_feats: utils.sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
            self.batch_labels: labels
        }
        return feed


    def predict(self, sess, adj_matrix, attr_matrix, alpha,
                nprop=2, ppr_normalization='sym', batch_size_logits=10000):


        local_logits = self._get_logits(sess, attr_matrix, adj_matrix.shape[0], batch_size_logits)
        logits = local_logits.copy()

        if ppr_normalization == 'sym':
            # Assume undirected (symmetric) adjacency matrix
            deg = adj_matrix.sum(1).A1
            deg_sqrt_inv = 1. / np.sqrt(np.maximum(deg, 1e-12))
            for _ in range(nprop):  # power iteration
                logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (deg_sqrt_inv[:, None] * logits)) + alpha * local_logits
        elif ppr_normalization == 'col':
            deg_col = adj_matrix.sum(0).A1
            deg_col_inv = 1. / np.maximum(deg_col, 1e-12)
            for _ in range(nprop):
                logits = (1 - alpha) * (adj_matrix @ (deg_col_inv[:, None] * logits)) + alpha * local_logits
        elif ppr_normalization == 'row':
            deg_row = adj_matrix.sum(1).A1
            deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
            for _ in range(nprop):
                logits = deg_row_inv_alpha[:, None] * (adj_matrix @ logits) + alpha * local_logits
        else:
            raise ValueError(f"Unknown PPR normalization: {ppr_normalization}")
        predictions = logits.argmax(1)
        return predictions

    def get_vars(self, sess):
        return sess.run(tf.trainable_variables())

    def set_vars(self, sess, new_vars):
        set_all = [
                var.assign(new_vars[i])
                for i, var in enumerate(tf.trainable_variables())]
        sess.run(set_all)


def train(sess, model, attr_matrix, train_idx, topk_train, labels, epoch, batch_size):
    
    if sp.issparse(attr_matrix):
        attr_matrix = SparseRowIndexer(attr_matrix)
    for i in range(0, len(train_idx), batch_size):
        if (i + batch_size)<=len(labels):
            feed_train = model.feed_for_batch_train(attr_matrix,
                                                    topk_train[train_idx[i:i + batch_size]],
                                                    labels[train_idx[i:i + batch_size]],
                                                    key=i)
            _, preds = sess.run([model.update_op, model.preds], feed_train)

        else:
            feed_train = model.feed_for_batch_train(attr_matrix,
                                                    topk_train[train_idx[len(labels)-batch_size:len(labels)]],
                                                    labels[train_idx[len(labels)-batch_size:len(labels)]],
                                                    key=i)

            _, preds = sess.run([model.update_op, model.preds], feed_train)

    return

def train_encoder(sess, model, attr_matrix, train_idx, topk_train, labels, epoch, batch_size):
    
    if sp.issparse(attr_matrix):
        attr_matrix = SparseRowIndexer(attr_matrix)
    
    for i in range(0, len(train_idx), batch_size):
        if (i + batch_size)<=len(labels):
            feed_train = model.feed_for_encoder_batch_train(attr_matrix,                                                           
                                                            [train_idx[i:i + batch_size]],
                                                            topk_train[train_idx[i:i + batch_size]],
                                                            labels[train_idx[i:i + batch_size]],
                                                            key=i)
            _, preds = sess.run([model.encoder_update_op, model.encoder_train_preds], feed_train)

        else:
            feed_train = model.feed_for_encoder_batch_train(attr_matrix,                                                           
                                                            [train_idx[len(labels)-batch_size:len(labels)]],
                                                            topk_train[train_idx[len(labels)-batch_size:len(labels)]],
                                                            labels[train_idx[len(labels)-batch_size:len(labels)]],
                                                            key=i)

            _, preds = sess.run([model.encoder_update_op, model.encoder_train_preds], feed_train)
        
    model.cached_train.clear()
    return