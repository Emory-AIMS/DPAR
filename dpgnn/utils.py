import os
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import metis
import networkx as nx
from .privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent
from .sparsegraph import load_from_npz


def sparse_feeder(M):
    # Convert a sparse matrix to the format suitable for feeding as a tf.SparseTensor
    M = M.tocoo()
    return np.vstack((M.row, M.col)).T, M.data, M.shape


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.n_columns]

        return sp.csr_matrix((data, indices, indptr), shape=shape)


def split_random(n, n_train):
    rnd = np.random.permutation(n)
    train_idx = np.sort(rnd[:n_train])
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_idx))
    return train_idx, test_idx


def get_data(dataset_path, privacy_amplify_sampling_rate):
    
    dataset_name = dataset_path.split('/')[-1]

    if dataset_name in ['amazon', 'facebook', 'reddit', 'physics']:
        return get_data_3(dataset_path, privacy_amplify_sampling_rate)

    if dataset_name in ['cora_ml', 'pubmed', 'ms_academic']:
        dataset_path += ".npz"

    g = load_from_npz(dataset_path)
    if dataset_path.split('/')[-1] in ['cora_full.npz']:
        g.standardize()
    

    # number of nodes and attributes
    n, d = g.attr_matrix.shape
    class_number = len(np.unique(g.labels))
    print("Loading {} graph with #nodes={}, #attributes={}, #classes={}".format(dataset_path.split('/')[-1], n, d,
                                                                                class_number))

    attr_matrix = g.attr_matrix

    # Generate Train Subgraph and Test Subgraph
    dense_adj_matrix = g.adj_matrix.toarray()
    dense_attr_matrix = attr_matrix.toarray()

    # split train/test graph
    train_adj_lists = [[] for _ in range(len(dense_adj_matrix))]
    for node_index in range(len(dense_adj_matrix)):
        train_adj_lists[node_index] = list(np.nonzero(dense_adj_matrix[node_index])[0])
    _, groups = metis.part_graph(train_adj_lists, 9, seed=0)
    test_idx = np.where(np.asarray(groups) == 4)[0]
    valid_idx = np.where(np.asarray(groups) == 5)[0]
    train_total_idx = np.setdiff1d(np.arange(n),  np.hstack([test_idx, valid_idx]))
    print(train_total_idx)

    # use subsamples from all train nodes for actual training (privacy amplification)
    train_total_idx = np.random.permutation(train_total_idx)
    train_idx = train_total_idx[:int(np.ceil(privacy_amplify_sampling_rate*len(train_total_idx)))]

    # generate train subgraph
    train_labels = g.labels[train_idx]
    train_adj_matrix = dense_adj_matrix[train_idx, :][:, train_idx]
    num_edges = sum(sum(train_adj_matrix))
    np.fill_diagonal(train_adj_matrix, 1)
    train_adj_matrix = sp.csr_matrix(train_adj_matrix)

    train_attr_matrix = dense_attr_matrix[train_idx, :]
    train_attr_matrix = sp.csr_matrix(train_attr_matrix)
    train_index = np.arange(len(train_idx))

    # generate test subgraph
    test_labels = g.labels[test_idx]
    test_adj_matrix = dense_adj_matrix[test_idx, :][:, test_idx]
    np.fill_diagonal(test_adj_matrix, 1)
    test_adj_matrix = sp.csr_matrix(test_adj_matrix)

    test_attr_matrix = dense_attr_matrix[test_idx, :]
    test_attr_matrix = sp.csr_matrix(test_attr_matrix)
    if sp.issparse(test_attr_matrix):
        test_attr_matrix = SparseRowIndexer(test_attr_matrix)
    test_index = np.arange(len(test_idx))

    return train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
           test_attr_matrix, test_index, n, class_number, d, num_edges


def get_data_3(dataset_path, privacy_amplify_sampling_rate):
    
    features = np.loadtxt(os.path.join(dataset_path, "node_features.csv"), delimiter=",")
    labels = np.loadtxt(os.path.join(dataset_path, "node_labels.csv"), delimiter=",", dtype=np.int32)
    total_rows = np.loadtxt(os.path.join(dataset_path, "senders.csv"), delimiter=",", dtype=np.int32)
    total_cols = np.loadtxt(os.path.join(dataset_path, "receivers.csv"), delimiter=",", dtype=np.int32)
    #test_rows = np.loadtxt(os.path.join(dataset_path, "test_rows.csv"), delimiter=",", dtype=np.int32)
    #test_cols = np.loadtxt(os.path.join(dataset_path, "test_cols.csv"), delimiter=",", dtype=np.int32)
    train_total_idx = np.loadtxt(os.path.join(dataset_path, "total_train_idx.csv"), delimiter=",", dtype=np.int32)
    test_idx = np.loadtxt(os.path.join(dataset_path, "test_idx.csv"), delimiter=",", dtype=np.int32)
    valid_idx = np.loadtxt(os.path.join(dataset_path, "valid_idx.csv"), delimiter=",", dtype=np.int32)

    # use subsamples from all train nodes for actual training (privacy amplification)
    train_total_idx = np.random.permutation(train_total_idx)
    train_idx = train_total_idx[:int(np.ceil(privacy_amplify_sampling_rate*len(train_total_idx)))]

    indices = np.ones(len(total_rows))
    adj_matrix = sp.csr_matrix(sp.coo_matrix((indices,(total_rows, total_cols)), shape=(features.shape[0], features.shape[0])))
    train_adj_matrix = adj_matrix[train_idx, :][:, train_idx]
    train_adj_matrix.setdiag(np.ones(len(train_idx)))
    test_adj_matrix = adj_matrix[test_idx, :][:, test_idx]
    test_adj_matrix.setdiag(np.ones(len(test_idx)))

    train_labels = labels[train_idx]
    train_attr_matrix = sp.csr_matrix(features[train_idx, :])
    train_index = np.arange(len(train_idx))
    test_labels = labels[test_idx]
    test_attr_matrix = sp.csr_matrix(features[test_idx, :])
    test_index = np.arange(len(test_idx))
    class_number=  len(np.unique(labels))
    n, d = features.shape
    num_edges = len(train_idx)

    

    return train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
           test_attr_matrix, test_index, n, class_number, d, num_edges


def get_data_2(dataset_path, privacy_amplify_sampling_rate):

    dataset_name = dataset_path.split('/')[-1]

    if dataset_name in ['ogbn_products']:
        _split_property = 'split/sales_ranking/'

        train_split_file = os.path.join(
            dataset_path, _split_property, 'train.csv.gz')
        test_split_file = os.path.join(
            dataset_path, _split_property, 'test.csv.gz')

        _features = np.loadtxt(os.path.join(dataset_path, 'raw/node-feat.csv.gz'), delimiter=',')
        _labels = np.loadtxt(os.path.join(dataset_path, 'raw/node-label.csv.gz'), delimiter=',')
        _edge = np.loadtxt(os.path.join(dataset_path, 'raw/edge.csv.gz'), delimiter=',')
        
        n, d = _features.shape
        class_number = len(np.unique(_labels))
        
        train_total_idx = np.loadtxt(train_split_file, delimiter=',', dtype=np.int32)
        test_idx = np.loadtxt(test_split_file, delimiter=',', dtype=np.int32)
        
        # use subsamples from all train nodes for actual training (privacy amplification)
        train_total_idx = np.random.permutation(train_total_idx)
        train_idx = train_total_idx[:int(np.ceil(privacy_amplify_sampling_rate*len(train_total_idx)))]
         
        _graph = nx.Graph()
        _graph.add_edges_from(_edge)
        _adj_matrix = nx.to_scipy_sparse_matrix(_graph)

        # Cut the graph
        _graph_train = _get_graph_for_split_with_self_loop(_adj_matrix, train_idx)
        _graph_test = _get_graph_for_split_with_self_loop(_adj_matrix, test_idx)

        # Generate train subgraph
        train_labels = _labels[train_idx]
        train_adj_matrix = nx.to_scipy_sparse_matrix(_graph_train)
        train_attr_matrix = sp.csr_matrix(_attr_matrix[train_idx, :])
        train_index = np.arange(len(train_idx))
        num_edges = _graph_train.number_of_edges()

        # Generate test subgraph
        test_labels = _labels[test_idx]
        test_adj_matrix = nx.to_scipy_sparse_matrix(_graph_test)
        test_attr_matrix = sp.csr_matrix(_attr_matrix[test_idx, :])
        if sp.issparse(test_attr_matrix):
            test_attr_matrix = SparseRowIndexer(test_attr_matrix)
        test_index = np.arange(len(test_idx))


    elif dataset_name in ['facebook']:
  
        _name = "facebook"
        _targets = ['status', 'gender', 'major', 'minor', 'housing', 'year']
        _data_path = os.path.join(dataset_path, "UIllinois20.mat")
        _data_dict = loadmat(_data_path)
        
        # Load graph
        _adj_matrix = sp.csr_matrix(_data_dict['A'])
        _attr_matrix = _data_dict['local_info'][:,:-1]
        _labels = _data_dict['local_info'][:,-1]
        
        n, d = _attr_matrix.shape
        class_number = len(np.unique(_labels))
        print("Loading {} graph with #nodes={}, #attributes={}, #classes={}".format(dataset_path.split('/')[-1], n, d,
                                                                                class_number))

        # Split train/test nodes
        _dense_adj_matrix = nx.from_scipy_sparse_matrix(_adj_matrix)
        _, groups = metis.part_graph(_dense_adj_matrix, 10, seed=0)
        test_idx = np.where(np.asarray(groups) == 4)[0]
        valid_idx = np.where(np.asarray(groups) == 5)[0]
        train_total_idx = np.setdiff1d(np.arange(_adj_matrix.shape[0]), np.hstack([test_idx, valid_idx]))

        # Use subsamples from all train nodes
        train_total_idx = np.random.permutation(train_total_idx)
        train_idx = train_total_idx[:int(np.ceil(privacy_amplify_sampling_rate*len(train_total_idx)))]

        # Cut the graph
        _graph_train = _get_graph_for_split_with_self_loop(_adj_matrix, train_idx)
        _graph_test = _get_graph_for_split_with_self_loop(_adj_matrix, test_idx)

        # Generate train subgraph
        train_labels = _labels[train_idx]
        train_adj_matrix = nx.to_scipy_sparse_matrix(_graph_train)
        train_attr_matrix = sp.csr_matrix(_attr_matrix[train_idx, :])
        train_index = np.arange(len(train_idx))
        num_edges = _graph_train.number_of_edges()

        # Generate test subgraph
        test_labels = _labels[test_idx]
        test_adj_matrix = nx.to_scipy_sparse_matrix(_graph_test)
        test_attr_matrix = sp.csr_matrix(_attr_matrix[test_idx, :])
        if sp.issparse(test_attr_matrix):
            test_attr_matrix = SparseRowIndexer(test_attr_matrix)
        test_index = np.arange(len(test_idx))

    elif dataset_name in ['reddit']:
    
        _graph_data_path = os.path.join(dataset_path, "raw", "reddit_graph.npz")
        _attr_data_path = os.path.join(dataset_path, "raw", "reddit_data.npz")
        _name = "reddit"

        # Load graph
        _adj_matrix = sp.csr_matrix(sp.load_npz(_graph_data_path))
        _data_matrix = np.load(_attr_data_path)
        _attr_matrix = _data_matrix['feature']
        _labels = _data_matrix['label']

        n, d = _attr_matrix.shape
        class_number = len(np.unique(_labels))
        print("Loading {} graph with #nodes={}, #attributes={}, #classes={}".format(dataset_path.split('/')[-1], n, d,
                                                                                class_number))
        
        # split train/test nodes
        train_total_idx = np.loadtxt(os.path.join(dataset_path, "processed", "reddit_train_total_idx.csv"), delimiter=',', dtype=np.int32)
        valid_idx = np.loadtxt(os.path.join(dataset_path, "processed", "reddit_valid_idx.csv"), delimiter=',', dtype=np.int32)
        test_idx = np.loadtxt(os.path.join(dataset_path, "processed", "reddit_test_idx.csv"), delimiter=',', dtype=np.int32)
        
        # Use subsamples from all train nodes
        train_total_idx = np.random.permutation(train_total_idx)
        train_idx = train_total_idx[:int(np.ceil(privacy_amplify_sampling_rate*len(train_total_idx)))]
        
        # Cut the graph
        _graph_train = _get_graph_for_split_with_self_loop(_adj_matrix, train_idx)
        _graph_test = _get_graph_for_split_with_self_loop(_adj_matrix, test_idx)
        
        # Generate train subgraph
        train_labels = _labels[train_idx]
        train_adj_matrix = nx.to_scipy_sparse_matrix(_graph_train)
        train_attr_matrix = sp.csr_matrix(_attr_matrix[train_idx, :])
        train_index = np.arange(len(train_idx))
        num_edges = _graph_train.number_of_edges()
        
        # Generate test subgraph
        test_labels = _labels[test_idx]
        test_adj_matrix = nx.to_scipy_sparse_matrix(_graph_test)
        test_attr_matrix = sp.csr_matrix(_attr_matrix[test_idx, :])
        if sp.issparse(test_attr_matrix):
            test_attr_matrix = SparseRowIndexer(test_attr_matrix)
        test_index = np.arange(len(test_idx))


    return train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
           test_attr_matrix, test_index, n, class_number, d, num_edges


def _get_graph_for_split_with_self_loop(adj_full, split_set):
  """Returns the induced subgraph for the required split."""
  def edge_generator():
    senders, receivers = adj_full.nonzero()
    for sender, receiver in zip(senders, receivers):
      if sender in split_set and receiver in split_set:
        yield sender, receiver
  
  def self_loop_generator():
    for idx in split_set:
      yield idx, idx

  graph_split = nx.Graph()
  graph_split.add_nodes_from(split_set)
  graph_split.add_edges_from(edge_generator())
  graph_split.add_edges_from(self_loop_generator())
  return graph_split


def compute_epsilon(steps, sigma, delta, sampling_rate):
    """Computes epsilon value for given hyper-parameters."""
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    rdp = compute_rdp(q=sampling_rate,
                      noise_multiplier=sigma,
                      steps=steps,
                      orders=orders)
    return get_privacy_spent(orders, rdp, target_delta=delta)[0]


def EM_Gumbel_Optimal(EM_eps, topk, svt_sensitivity, report_noise_val_eps, p_vector):
    p_vector_copy = np.copy(p_vector)
    p_vector_copy_2 = np.copy(p_vector)
    # clip
    for idx in range(len(p_vector)):
        ppr_value_ = p_vector_copy[idx]
        ppr_value_2 = p_vector_copy_2[idx]
        if ppr_value_ > svt_sensitivity:
            p_vector_copy[idx] = svt_sensitivity
        if ppr_value_2 > svt_sensitivity:
            p_vector_copy_2[idx] = svt_sensitivity

    gumbel_noise = np.random.gumbel(scale=2 * svt_sensitivity / EM_eps, size=p_vector.shape)
    p_vector_copy += gumbel_noise
    j_em = np.argsort(p_vector_copy)[-topk:]

    val_em = []
    for j in j_em:
        if report_noise_val_eps != 0:
            val = p_vector_copy_2[j] + np.random.laplace(scale=topk * svt_sensitivity / report_noise_val_eps)
        else:
            val = 1.0 / topk
        val_em.append(val)
    return list(j_em), val_em

