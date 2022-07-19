#%%
import matplotlib.pyplot as plt

fig_width_pt = 455.24411  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = ((5**.5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'pdf',
          'savefig.format': 'pdf',
          'figure.figsize': (16,9),
          'figure.dpi': 100,
          'font.family':'serif',
          'font.size':14,
          'image.cmap':"hot_r"}


plt.rcParams.update(params)
# %%
