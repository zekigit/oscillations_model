import mne
import math
import numpy as np
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter

subjects = ['s1', 's2', 's3']

study_path = '/Users/lpen/Documents/wake_sleep/study/'
ref = 'avg'  # reference: 'avg' (average) or 'bip' (bipolar)
subject = 's1'

# Prepare data --------
# load
file_name = '%s_wake_%s_16s-epo.fif' % (subject, ref)
file_path = op.join(study_path, subject, 'data', 'epochs', file_name)
epochs = mne.read_epochs(file_path)

# apply filter
bandpass = [90, 95]
epochs.filter(bandpass[0], bandpass[1], fir_design='firwin')

# get data
data_mat = epochs.get_data()  # 3D matrix: epochs x channels x samples
sfreq = int(epochs.info['sfreq'])

pd.set_option('display.expand_frame_repr', False)  # to show channel info in wide format
ch_info = pd.read_csv(op.join(study_path, 'varios', 'chan_info_all.csv'))
subj_ch_info = ch_info.loc[(ch_info.Subject == subject) & (ch_info.Condition == 'wake')]
print(subj_ch_info)

# rename channels and select those in white matter
old_names = epochs.info['ch_names']
new_names = {w: w.replace("'", "_i") for w in old_names}  # correct difference in ch names btw ch_info and epochs variables
epochs.rename_channels(new_names)
epochs.pick_channels(subj_ch_info['channel'].tolist())


# Compute phase and phase diff and plot for 2 channels --------
# select 2 channels and window length (seconds)
ch1, ch2 = 1, 20
win = 1

t_win = int(sfreq * win)
t = np.linspace(0, win, t_win)  # create time vector

sig_1 = data_mat[ch1, 20, :]
sig_2 = data_mat[ch2, 20, :]

signals = [sig_1, sig_2]
asigs = list()

# plot
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 10))
grid = plt.GridSpec(5, 2)
ax0 = plt.subplot(grid[0, 0:])
ax1 = plt.subplot(grid[1, 0:], sharex=ax0)
ax2 = plt.subplot(grid[2, 0:], sharex=ax0)
ax3 = plt.subplot(grid[3, 0:], sharex=ax0)
ax4 = plt.subplot(grid[4, 0], polar=True)
ax5 = plt.subplot(grid[4, 1])
colors = ['green', 'orange']

ax0.grid()
ax1.grid()
ax2.grid()
ax3.grid()

for ix_s, s in enumerate(signals):
    s = s[:t_win]
    asig = hilbert(s) # hilbert transform
    asigs.append(asig)

    ax0.plot(t, s, colors[ix_s])  # , linestyle='-', marker='o', alpha=0.7, markersize=5
    ax0.hlines(0, 0, win)
    ax1.plot(t, np.angle(asig), colors[ix_s])

legend = ax0.legend([epochs.info['ch_names'][ch1], epochs.info['ch_names'][ch2]], shadow=True)
ph_dif = np.angle(asigs[0]) - np.angle(asigs[1])

ax2.plot(t, ph_dif, 'w') # , linestyle='-', marker='o', alpha=0.7, markersize=5
ax2.hlines(0, 0, win)

abs_dif = abs(np.angle(asigs[0])) - abs(np.angle(asigs[1]))
ax3.plot(t, abs(abs_dif), 'y', alpha=0.7)
#ax3.hlines(0, 0, win, color='w', linestyle='--')
ax3.set_xlabel('Time', multialignment='left')

#asig_dif = hilbert(abs_dif)
filt_dif = savgol_filter(abs(abs_dif), 21, 3)
ax3.plot(t, filt_dif, 'c')

mean_difs = np.array([np.mean((np.exp(1j*np.angle(asigs[0][ix])), np.exp(1j*np.angle(asigs[1][ix])))) for ix in range(len(asigs[0]))])
vecs_l = np.abs(mean_difs)
vecs_ph = np.angle(mean_difs)
mean_vec = np.mean(np.exp(1j * vecs_ph))
mean_vec_l = np.abs(mean_vec)
mean_vec_ph = np.angle(mean_vec)

m_vec = np.mean(math.e**(1j*(np.angle(asigs[0]) - np.angle(asigs[1]))))
m_vec_l = np.abs(m_vec)
m_vec_ph = np.angle(m_vec)


# for l, p in zip(m_vec, vec_ph):
#     ax4.plot([0, p], [0, l], alpha=0.5)

# ax4.hist(vec_ph)
ax4.plot([0, m_vec_ph], [0, m_vec_l], color='r', lw=5)
ax4.set_rmax(0.5)

h = ax5.hist(vecs_l, bins=72, color='r')
ax5.set_xlabel('vector length')
ax5.set_ylabel('count')

for ax, lab in zip([ax0, ax1, ax2, ax3], ['Amplitude', 'Phase', 'Phase dif', 'Abs dif']):
    ax.set_ylabel(lab)
fig.suptitle('Bandpass = {} - {} Hz' .format(bandpass[0], bandpass[1]))

plt.tight_layout(pad=0.2)
plt.savefig('phase_panel.eps', format='eps', dpi=300)

# phase = np.angle(asigs[0])
# a= np.mean(np.exp(1j*phase))
#
# a=asigs[0][10]
# b=np.pi/2
#
# c=np.exp(1j*a)
#
a = asigs[0][10]
b = asigs[1][10]
c = np.mean((a, b))

a1 = np.exp(1j*np.angle(a))  # to unit vector
b1 = np.exp(1j*np.angle(b))
c1 = np.mean((a1, b1))


for v, co in zip([a1, b1, c1], ['r', 'b', 'orange']):
    plt.polar([0, np.angle(v)], [0, np.abs(v)], color=co, linewidth=4)
plt.savefig('phase_diff.eps', format='eps', dpi=300)


# r = np.arange(0, 2, 0.1)
# theta = 2 * np.pi * r
#
# ax = plt.subplot(111, projection='polar')
# ax.plot(ph_dif, np.repeat(1,len(ph_dif)), 'o')
# ax.set_rmax(2)
# ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
# ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
# ax.grid(True)
