import numba
import numpy as np
import scipy.sparse as sp
from . import utils


def construct_sparse(neighbors, weights, shape):
    i = np.repeat(np.arange(len(neighbors)), np.fromiter(map(len, neighbors), dtype=np.int))
    j = np.concatenate(neighbors)
    return sp.coo_matrix((np.concatenate(weights), (i, j)), shape)


# clip and add noise to the output of pr
@numba.njit(cache=True)
def calc_ppr_node_ista_p_dp(nnodes, alpha, epsilon, sigma, clip_bound, rho, out_degree, node_index, deg_inv, indices,
                           indptr, dp_ppr):
    s_vector = np.zeros(nnodes)
    s_vector[node_index] = 1

    p_vector_old = np.zeros(nnodes)
    p_vector_new = np.zeros(nnodes)

    delta_fp_old = np.zeros(nnodes)
    delta_fp_old[node_index] = -alpha * deg_inv[node_index]
    delta_fp_shape = delta_fp_old.shape

    while np.linalg.norm(delta_fp_old, np.inf) > (1 + epsilon) * rho * alpha:
        S_k = np.where(p_vector_old - delta_fp_old >= rho * alpha)[0]
        Sk_list = list(S_k)
        delta_pk = -(delta_fp_old[S_k] + rho * alpha)

        p_vector_new[S_k] = p_vector_old[S_k] + delta_pk

        delta_fp_new = delta_fp_old

        Is_delta_pk = np.zeros(nnodes)
        Is_delta_pk[S_k] = delta_pk
        for i in S_k:
            tmp_sum = 0
            for l in indices[indptr[i]:indptr[i + 1]]:
                if l in Sk_list:
                    tmp_sum += Is_delta_pk[l] / out_degree[l]

            delta_fp_new[i] = (1 - 1. / out_degree[i]) * delta_fp_old[i] - \
                              rho * alpha / out_degree[i] - \
                              0.5 * (1 - alpha) * Is_delta_pk[i] / out_degree[i] - \
                              0.5 * (1 - alpha) / out_degree[i] * tmp_sum

        neighbors_set = []
        for s in S_k:
            for neighbor in indices[indptr[s]:indptr[s + 1]]:
                if neighbor not in neighbors_set:
                    if neighbor not in Sk_list:
                        neighbors_set.append(neighbor)
        for j in neighbors_set:
            tmp_sum = 0
            for l in indices[indptr[j]:indptr[j + 1]]:
                if l in Sk_list:
                    tmp_sum += Is_delta_pk[l] / out_degree[l]

            delta_fp_new[j] = delta_fp_old[j] - 0.5 * (1 - alpha) / out_degree[j] * tmp_sum

        delta_fp_old = delta_fp_new
        p_vector_old = p_vector_new

    if dp_ppr:
        gaussian_noise = np.random.normal(loc=0.0, scale=sigma, size=delta_fp_shape)
        p_vector_old_norm = np.linalg.norm(p_vector_old)
        if p_vector_old_norm > clip_bound:
            p_vector_old_clip = clip_bound / p_vector_old_norm * p_vector_old
        else:
            p_vector_old_clip = p_vector_old
        p_vector_old_tilde = p_vector_old_clip + gaussian_noise
    else:
        p_vector_old_tilde = p_vector_old

    return p_vector_old_tilde


def ppr_topk_ista_helper(nnodes, nodes, alpha, epsilon, sigma, clip_bound_ista, rho, out_degree, deg_inv, indices,
                         indptr, dp_ppr, topk, em_sensitivity, report_noise_val_eps,
                         EM, EM_eps):
    js = [np.zeros(0, dtype=np.int64)] * nnodes
    vals = [np.zeros(0, dtype=np.float64)] * nnodes

    count = 0
    for i in numba.prange(len(nodes)):
        node_index = nodes[i]
        p_vector = calc_ppr_node_ista_p_dp(nnodes, alpha, epsilon, sigma, clip_bound_ista, rho, out_degree,
                                           node_index, deg_inv, indices, indptr, dp_ppr)

        if EM:
            j_dp, val_dp = utils.EM_Gumbel_Optimal(EM_eps, topk, em_sensitivity, report_noise_val_eps, p_vector)
            js[count] = np.asarray(j_dp)
            vals[count] = np.asarray(val_dp)
            count += 1
        else:
            j = np.nonzero(p_vector)[0]
            val = p_vector[j]
            idx_topk = np.argsort(val)[-topk:]
            js[count] = j[idx_topk]
            vals[count] = val[idx_topk]
            count += 1
    return js, vals


def ppr_topk_ista(adj_matrix, alpha, epsilon, rho, nodes, topk, dp_ppr, sigma, clip_bound_ista, em_sensitivity,
                  report_noise_val_eps, EM, EM_eps):
    """Calculate the PPR matrix approximately using ISTA """

    out_degree = np.sum(adj_matrix > 0, axis=1).A1
    nnodes = adj_matrix.shape[0]

    deg = adj_matrix.sum(1).A1
    deg_inv = 1. / np.maximum(deg, 1e-12)

    js, vals = ppr_topk_ista_helper(nnodes, nodes, alpha, epsilon, sigma, clip_bound_ista, rho, out_degree, deg_inv,
                                    adj_matrix.indices, adj_matrix.indptr, dp_ppr, topk, em_sensitivity,
                                    report_noise_val_eps, EM, EM_eps)

    return construct_sparse(js, vals, (len(nodes), nnodes))


def topk_ppr_matrix_ista(adj_matrix, alpha, eps, rho, idx, topk, sigma, clip_bound_ista, dp_ppr,
                         em_sensitivity, report_noise_val_eps, EM, EM_eps):
    """Create a sparse matrix where each node has up to the topk PPR neighbors and their weights."""

    topk_matrix = ppr_topk_ista(adj_matrix, alpha, eps, rho, idx, topk, dp_ppr, sigma, clip_bound_ista,
                                em_sensitivity, report_noise_val_eps, EM, EM_eps).tocsr()

    return topk_matrix
