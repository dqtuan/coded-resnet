"""
    author: Tuan Dinh
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 14
plt.style.use('ggplot')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE, titleweight='bold')     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE, labelweight='bold')    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('text', usetex=True)

class Plotter:
    @staticmethod
    def plot_loss(train, title, savepath, test=None):
        fig = plt.figure()
        plt.plot(np.arange(len(train)), train, label='train', c='r')
        if not(test is None):
            plt.plot(np.arange(len(test)) * 10 + 1, test, label='test', c='b')

        plt.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        # plt.show()
        fig.tight_layout()
        fig.savefig(savepath, bbox_inches='tight')

    @staticmethod
    def plot_lines():
        linestyles = ['-', '--', '-.', ':']
        i = 0
        numpoints = 10
        acc = np.linspace(0.35, 0.88, numpoints)
        recon_b = np.linspace(0.6, 0.7, numpoints)
        recon_b_hat_50 = np.linspace(0.5, 0, numpoints)
        recon_b_hat_30 = np.linspace(0.55, 0.3, numpoints)
        spectral_norm = np.linspace(1.5, 1, numpoints)

        fig = plt.figure()
        plt.plot(range(numpoints), acc, label=r'Accuracy', c='b')
        plt.plot(range(numpoints), recon_b, label=r'$B$', c='g')
        plt.plot(range(numpoints), recon_b_hat_50, label=r'$B_{50}$', c='y')
        plt.plot(range(numpoints), recon_b_hat_30, label=r'$B_{30}$', c='k')
        plt.plot(range(numpoints), spectral_norm, label=r'$\sigma$', c='r')

        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel("Value")
        plt.title('Result')
        # plt.show()
        fig.tight_layout()
        fig.savefig('spectralnorm.jpg', bbox_inches='tight')

    @staticmethod
    def plot_tnse(data, labels, img_path, num_classes=3):
        '''
            data: [N, d]
            labels: [d]
        '''
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        result = tsne.fit_transform(data)
        vis_x = result[:, 0]
        vis_y = result[:, 1]

        indices = np.argwhere(np.abs(result[:, 0]) < 10)
        indices = indices.reshape(indices.shape[0])
        result = result[indices, :]
        labels = labels[indices]

        indices = np.argwhere(np.abs(result[:, 1]) < 10)
        indices = indices.reshape(indices.shape[0])
        result = result[indices, :]
        labels = labels[indices]
        vis_x = result[:, 0]
        vis_y = result[:, 1]
        # plot
        colors = ['y', 'k', 'r', 'g', 'b',  'm', 'c']
        fig = plt.figure(figsize=(10,10))
        for k in range(num_classes):
               plt.scatter(vis_x[labels == k], vis_y[labels == k], c=colors[k], label=k)
        plt.legend()
        # plt.show()
        fig.savefig(img_path)
        # dict_tsne = {}
        # dict_tsne['result'] = result
        # np.save(tsne_name, dict_tsne)
