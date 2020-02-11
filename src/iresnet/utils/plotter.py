

from matplotlib import pyplot as plt
import numpy as np

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
