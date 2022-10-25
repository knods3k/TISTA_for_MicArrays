#%%
from os.path import normpath, join

from tools.prediction_models import tista, cmf 
from tools.training.data import DataGenerator
from tools.physical import PhyiscalModel, INCREMENT, NMICS
from tools.pyplot_setup import params

import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from acoular import L_p

IM_KWARGS = {"origin":"lower", "extent":[-.5,.5,-.5,.5],"vmax":95,"vmin":75, "interpolation":None}


T = 60
NSOURCES = 10

#%%
all = []
for HE in [4, 16]:
    for model_key in ['TISTA', 'CMF']:
        physics = PhyiscalModel(HE, INCREMENT, NMICS)
        random_matrix_path = normpath(join("data", "random_matrices", f"{HE}"))
        As = random_matrix_path
        generator = DataGenerator(As, 1, NSOURCES)
        data = generator.get_batch()

        y,x = next(iter(data))
        
        if model_key == 'TISTA':
            smap = tista(y, physics, T=T)
        if model_key == 'CMF':
            smap = cmf(y, physics, max_iter=T)

        df = pd.DataFrame([{'Model': model_key, 'He': HE, 'Sourcemap': smap}])
        all.append(df)
#%%
df = pd.DataFrame([{'Model': 'True', 'He': HE, 'Sourcemap': physics.vector_to_sourcemap(x).numpy()}])
all.insert(2,df)

sourcemaps = pd.concat(all, ignore_index=True)

#%%

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(2,3), axes_pad=.1)#,\
    #  cbar_mode='each', cbar_pad='0%', cbar_location='bottom')

for i, ax in enumerate(grid):
    cax = grid.cbar_axes[i]
    if i == 5:
        ax.axis('off')
        cax.axis('off')
        cax = fig.add_axes([.63,.29,.2,.03])
        cbar = plt.colorbar(im, cax=cax, shrink=.1, orientation='horizontal')
        cbar.set_label('Sound Pressure Level [dB]')
        cbar.set_ticks([75, 80, 85, 90, 95])
        break
    # if i == 2:
    #     cbar = plt.colorbar(im, cax=cax, shrink=.1, orientation='horizontal')
    #     cbar.set_label('Sound Pressure Level [dB]')
    #     cbar.set_ticks([80, 85, 90])
    # else:
    #     cax.axis('off')

    d = sourcemaps.loc[i]
    model = d['Model']
    He = d['He']
    smap = d['Sourcemap']
    smap = L_p(smap)
    im = ax.imshow(smap, **IM_KWARGS)
    ax.set_xticks([])
    ax.set_yticks([])
    if i <= 2:
        ax.set_title(model)
    if i == 0 or i == 3:
        ax.set_ylabel(f'He={He}')

plt.savefig('data/plots/smaps.pdf')

# %%
