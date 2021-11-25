#%%
from bernoulli_gaussian import get_bg_batch
from jahnke import create_sensing_matrix
from numpy import einsum, mean, angle, abs, std, linspace
from scipy.stats import norm
from config import HE, SNR, mg, sv
from _jahnke_reverse import unstack_complex_vector
from matplotlib import pyplot as plt
from tensorflow import einsum as tfeinsum

# SETUP
freq = HE*343
A = create_sensing_matrix(sv,mg,freq)
SNR = 20

BINS = 100
NCASES = 10**6

fig_mag,axs_mag = plt.subplots(1,1,figsize=(16,9))
fig_ang,axs_ang = plt.subplots(1,1,figsize=(16,9))

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

mag_mean = mean(y_calc_mag)
mag_std = std(y_calc_mag)

diff_mag = y_simu_mag - y_calc_mag
diff_mag = -diff_mag
diff_mag_mean = mean(diff_mag)
diff_mag_std = std(diff_mag)

mag_x = linspace(mag_mean - mag_std**2,mag_mean + mag_std**2, BINS)
mag_norm = norm(loc=mag_mean,scale=mag_std).pdf(mag_x)

axs_mag.hist(diff_mag,bins=BINS)
axs_mag.set_title(f"Magnitude Histogram | SNR = {SNR} | Mean = "+"{:.2}".format(diff_mag_mean) +" | Std. Deviation = "+"{:.2}".format(diff_mag_std))
axs_mag.set(xlabel="Magnitude [${Pa}^2$]",ylabel="Number of Cases")


# ANGLE
y_simu_ang = angle(y_simu,deg=True)
y_calc_ang = angle(y_calc,deg=True)

ang_mean = mean(y_simu_ang)
ang_std = std(y_simu_ang)

diff_ang = y_simu_ang - y_calc_ang
diff_ang_mean = mean(diff_ang)
diff_ang_std = std(diff_ang)

ang_x = linspace(-1,1, BINS)
ang_norm = norm(loc=ang_mean,scale=ang_std).pdf(ang_x)

axs_ang.hist(diff_ang,bins=BINS)
axs_ang.set_title(f"Phase Histogram | SNR = {SNR} | Mean = "+"{:.2}".format(ang_mean) +" | Std. Deviation = "+"{:.2}".format(ang_std))
axs_ang.set(xlabel="Phase [°]",ylabel="Number of Cases")


# SHOW
fig_mag.show()
fig_ang.show()

#%%
fig_mag.savefig(f"../plots/histogram_magnitude_SNR={SNR}_NCASES={NCASES}.png")
#fig_ang.savefig(f"../plots/histogram_phase_SNR={SNR}_NCASES={NCASES}.png")




#%%

    ax = diff_mag.plot.hist(bins=999,title="Magnitude",legend=False,figsize=(16,9))
    fig = ax.get_figure()
    fig.savefig("../plots/Deviation_magnitude_SNR={SNR}.png")

    ax = diff_ang.plot.hist(bins=999,title="Phase",legend=False,figsize=(16,9))
    fig = ax.get_figure()
    fig.savefig(f"../plots/Deviation_phase_SNR={SNR}.png")

# %%

# %%
