import numpy as np

def load_sample_from_npy(file_path, return_cupy, dtype):
    data = np.load(file_path)
    data = data.astype(dtype)
    if return_cupy:
        import cupy as cp
        data = cp.array(data)
    return data
