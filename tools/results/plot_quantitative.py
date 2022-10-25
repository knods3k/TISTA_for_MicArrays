#%%
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler

from tools.pyplot_setup import params

converge = pd.read_pickle('data/convergence.pkl')
nsources = pd.read_pickle('data/nsources.pkl') 

#%%
default_cycler = (
    cycler(color=['k', 'k', 'k', 'k']) +
    cycler(linestyle=['-', '-', '--', '--']) +
    cycler(marker=['x','o','x','o'])
)
plt.rc('axes', prop_cycle=default_cycler)

legend_elements = [
    Line2D([0],[0], linestyle='--', color='k', label='CMF'),
    Line2D([0],[0], linestyle='-', color='k', label='TISTA'),
    Line2D([0],[0], marker='x', markerfacecolor='k', markeredgecolor='k', color='w', label='He=4'),
    Line2D([0],[0], marker='o', markerfacecolor='k', markeredgecolor='k', color='w', label='He=16')
]


plt.figure()
plt.subplot(121)
plt.ylabel(r'$\tilde{\mathcal{L}}$')
plt.plot(nsources)
plt.yscale('log')
b,t = plt.ylim()
plt.xlabel('Number of sources')
plt.title('A')

ax = plt.subplot(122)
plt.ylabel('      ')
plt.plot(converge)
plt.yscale('log')
plt.ylim(b,t)
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.xlabel('Number of iterations')
plt.title('B')
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('data/plots/conv_nsources.pdf')

# %%
