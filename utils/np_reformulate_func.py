import numpy as np


def partition_avg(np_array: np.ndarray, batch_size: int):
    if batch_size > np_array.size:
        return
    avg_array_size = int(np_array.size / batch_size)
    trunc_array = np_array[:avg_array_size * batch_size]
    partition_avg_array = np.average(np.reshape(trunc_array, newshape=(-1, batch_size)), axis=1)
    return np.linspace(1, partition_avg_array.size, partition_avg_array.size), partition_avg_array


print(partition_avg(np.ones(10),3))

