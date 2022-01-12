#%%
from numpy import einsum, mean, angle, abs, std, linspace
from matplotlib import pyplot as plt
plt.rc("font",size=14)
from tensorflow import einsum as tfeinsum
from scipy.stats import norm
import pandas as pd

from tools.training.data import get_bg_batch
from tools.jahnke import create_sensing_matrix
from tools.environment import HE, SNR, mg, sv
from tools.model import A
from tools.scratchfiles._jahnke_reverse import unstack_complex_vector

# SETUP
freq = HE*343
SNR = 40

BINS = 100
NCASES = 10**5

dict = {}

for SNR in [40,20,10]:
    data = get_bg_batch(A,NCASES,SNR=SNR)

    y,x = next(iter(data))


    # NOISY CASE
    y_simu = y.numpy()
    y_simu = unstack_complex_vector(y_simu.T).T
    y_simu = y_simu[:,0]


    # NOISELESS CASE
    y_calc = tfeinsum("ij,kj->ki",A,x).numpy()
    y_calc = unstack_complex_vector(y_calc.T).T
    y_calc = y_calc[:,0]


    # MAGNITUDE
    y_simu_mag = abs(y_simu)
    y_calc_mag = abs(y_calc)
    diff_mag   = y_simu_mag - y_calc_mag


    # ANGLE
    y_simu_ang = angle(y_simu,deg=True)
    y_calc_ang = angle(y_calc,deg=True)
    diff_ang   = y_simu_ang - y_calc_ang

    dict[f"Magnitude {SNR}"]    = diff_mag
    dict[f"Phase {SNR}"]        = diff_ang

df = pd.DataFrame(dict)
df.describe()


#%%
# PANDAS
df = pd.DataFrame({"Magnitude": diff_mag, "Phase": diff_ang})
df.describe()
ax1, ax2 = df.hist(bins=BINS,sharey=True,)[0]
ax1.set_title("")
ax2.set_title("")
ax1.set_xlabel(r"Magnitude Deviation [${Pa}^2$]")
ax2.set_xlabel(r"Phase Deviation [°]")
ax1.set_ylabel("Statistical Frequency")

ax1.figure.savefig(f"data/plots/phase_mag_hist_snr{SNR}.pdf")

#%%
fig_mag.savefig(f"../plots/histogram_magnitude_SNR={SNR}_NCASES={NCASES}.png")
#fig_ang.savefig(f"../plots/histogram_phase_SNR={SNR}_NCASES={NCASES}.png")



