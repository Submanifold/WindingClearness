import numpy as np
from tqdm import tqdm

def construct_A_2d(sample_pts, query_pts, w_x):

    num_query_pts = query_pts.shape[0]
    A = query_pts[:, None] - sample_pts[None, :] # [N_query, N_sample, 2]
    dist = np.sqrt((A ** 2).sum(-1))   # [N_query, N_sample], sqrt(|x^i-y^j|^2)
    inv_dist = np.where(dist > w_x, 1 / dist, 1 / w_x)
    inv_cub_dist = inv_dist ** 2 / 2 / np.pi  # [N_query, N_sample]

    A *= inv_cub_dist[..., None]  # [N_query, N_sample, 2]
    A = A.transpose((0, 2, 1))
    A = A.reshape(num_query_pts, -1)

    return A


def construct_A(sample_pts, query_pts, w_x):
    num_query_pts = query_pts.shape[0]
    A = query_pts[:, None] - sample_pts[None, :] # [N_query, N_sample, 3]
    dist = np.sqrt((A ** 2).sum(-1))   # [N_query, N_sample], sqrt(|x^i-y^j|^2)
    inv_dist = np.where(dist > w_x, 1 / dist, 1 / w_x)  # / 4 / cp.pi
    inv_cub_dist = inv_dist ** 3 / 4 / np.pi  # [N_query, N_sample]

    A *= inv_cub_dist[..., None]  # [N_query, N_sample, 3]
    A = A.transpose((0, 2, 1))
    A = A.reshape(num_query_pts, -1)

    return A


def construct_A2(sample_pts, query_pts, w_x):
    num_query_pts = query_pts.shape[0]
    A = query_pts[:, None] - sample_pts[None, :] # [N_query, N_sample, 3]
    dist = np.sqrt((A ** 2).sum(-1))   # [N_query, N_sample], sqrt(|x^i-y^j|^2)
    dist = np.where(np.abs(dist) < 1e-6, 1e-6, dist)
    inv_dist = 1.0 / dist  # / 4 / cp.pi
    inv_cub_dist = inv_dist ** 3 / 4 / np.pi  # [N_query, N_sample]

    A *= inv_cub_dist[..., None]  # [N_query, N_sample, 3]
    A = A.transpose((0, 2, 1))
    A = A.reshape(num_query_pts, -1)
    A = np.clip(A, -1e5, 1e5)

    return A


def construct_b(query_dist_ms, w_x):
    b = np.clip(query_dist_ms, -w_x / 2, w_x / 2) / w_x + 1/2
    return b


def cal_indicator(sample_pts, number_sample_pts, query_pts, num_query_pts, mu, w_x, chunk_size):
    query_chunks = np.array_split(query_pts, num_query_pts // chunk_size + 1)
    query_vals = []

    tqdmbar_query = tqdm(list(query_chunks))

    for chunk in tqdmbar_query:
        chunk = np.array(chunk)
        A_show = construct_A(sample_pts, chunk, w_x)
        query_vals.append(np.matmul(A_show, mu))

    query_vals = np.concatenate(query_vals, axis=0)
    return query_vals

def cal_indicator2(sample_pts, number_sample_pts, query_pts, num_query_pts, mu, w_x, chunk_size):
    query_chunks = np.array_split(query_pts, num_query_pts // chunk_size + 1)
    query_vals = []

    tqdmbar_query = tqdm(list(query_chunks))

    for chunk in tqdmbar_query:
        chunk = np.array(chunk)
        A_show = construct_A2(sample_pts, chunk, w_x)
        query_vals.append(np.matmul(A_show, mu))

    query_vals = np.concatenate(query_vals, axis=0)
    return query_vals


#solving (A^TA + (rho - 1) * diag(A^T, A)) x = A^T b
def cg(A, rho, rhs, ep=1e-12, max_iter=50):
    x = np.zeros((A.shape[1]))
    r = p = rhs
    diag_ATA = np.diag(np.sum(np.power(A, 2), axis=0))
    AT = A.T
    for i in range(max_iter):
        r_norm_sq = np.dot(r, r)
        print(r_norm_sq)
        if (np.sqrt(r_norm_sq) < ep):
            return x
        ap = np.matmul(A, p)
        cp = np.matmul(diag_ATA, p)
        ap_norm_sq = np.dot(ap, ap)
        cp = np.dot(p, cp)
        de = ap_norm_sq + (rho - 1.0) * cp
        if de < 1e-8:
            de = 1e-8
        alpha = r_norm_sq / de
        x = x + alpha * p
        atap = np.matmul(AT, ap)
        r_plus1 = r - alpha * atap - alpha * (rho - 1.0) * cp
        r_plus1_norm_sq = np.dot(r_plus1, r_plus1)
        beta = r_plus1_norm_sq / r_norm_sq
        r = r_plus1
        p = r + beta * p
    return x