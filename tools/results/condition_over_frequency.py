#%%
import numpy as np
import matplotlib.pyplot as plt
from tools.physical import PhyiscalModel
from tools.pyplot_setup import params

plt.rcParams.update(params)

#%%
FREQ = 343 * 4
RES = 10

x = np.linspace(0,2000,RES)
y = np.empty(x.shape)
# z = np.empty(x.shape)

for j in range(RES):
	f = x[j]
	he = f /343
	p = PhyiscalModel(he, 0.01, 64)
	cond = np.linalg.cond(p.A)
	print(f"He = {np.round(he)} \t\t\t Condition = {cond}")
	y[j] = cond
	# cond_stack = np.linalg.cond(stack_complex_matrix(transform_rg_mg_to_A(rg,mg,f,i)))
	# z[j] = cond_stack

x /= 343

#%%

fig, ax = plt.subplots(1)
plt.xlabel("Helmholtz Number")
plt.ylabel("Condition Number")
ax.semilogy(x,y,"r")
# plt.axvline(x=4,ymax=.37,color="k",linestyle="--")
# plt.axvline(x=8,ymax=.08,color="k",linestyle="--")
# plt.axvline(x=16,ymax=.045,color="k",linestyle="--")
# ax.semilogy(x,z,"k--")
fig.show()

# %%

fig.savefig("plots/cond_he.pdf")
# %%
