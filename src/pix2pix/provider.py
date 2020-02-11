
import os
import numpy as np

data_path = '../results/fusion'
for k in range(389):
    fpath = os.path.join(data_path, "data_{}.npy".format(k))
    data = np.load(fpath, allow_pickle=True)
    data = data.item()
    if k == 0:
        data_a = data['a']
        data_b = data['b']
        data_ab = data['fusion']
    data_a = np.concatenate([data_a, data['a']], axis=0)
    data_b = np.concatenate([data_b, data['b']], axis=0)
    data_ab = np.concatenate([data_ab, data['fusion']], axis=0)

from IPython import embed; embed()
