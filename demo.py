# -*- coding: utf-8 -*-
"""
This file contains a demonstration of the computatin of the CosMIC metric score.

Author: Stephanie Reynolds
Supervisors: Pier Luigi Dragotti, Simon R Schultz
Date: 3rd July 2017
"""

import cosmic
import numpy as np
import matplotlib.pyplot as plt
import random

# set random seed for reproducible results
random.seed(1)
np.random.seed(1)

#%% generate fluorescence signal

#pulse parameters
alpha = 3.18
gamma = 34.49
A     = 1

# time stamps
T       = 0.08
t_start = 0
t_end   = 30
t       = np.arange(t_start, t_end, T)

# spike times
spike_rate = 1 #Hz
t_k        = np.random.exponential(spike_rate, t_end * spike_rate * 3)
t_k        = np.cumsum(t_k) 
t_k        = t_k[t_k < t_end]
K          = len(t_k)
a_k        = A * np.ones(K)

# simulate fluorescence signal
f       = cosmic.simulate_fluorescence_signal(t, t_k, a_k)

# noise level and noisy signal
sigma   = 0.25
noise   = np.random.normal(0, sigma, t.size)
f_noisy = f + noise

#%% simulate spike estimates

# spike estimates are distributed around a subset of the true spikes
K_hat  = int(np.rint(0.8 * K))
idx    = random.sample(list(range(K)), K_hat)

# add jitter to the spike estimates
jitter = 0.03
tt_k   = np.random.normal(t_k[idx], jitter)


#%% Compute CRB

# get crb from paramters
crb       = cosmic.compute_crb(T, A, sigma**2, alpha, gamma)

# get metric width from crb
width     = cosmic.compute_metric_width(crb)

# get metric score
[cos_score, cos_prec, cos_call,y,y_hat, t_y] = cosmic.compute_score(width, t_k, tt_k)

print("CosMIC score is: {}. Precision is: {}. Recall is: {}.".format(round(cos_score,3), round(cos_prec,3), round(cos_call,3)))

#%% plot results

# format figures
est_col = np.divide([147, 49, 87], 255)
true_col  = np.divide([108, 154, 51], 255)          
fig, ax  = plt.subplots(3, sharex=True)
plt.xlabel('Time (s)', fontsize = 16)

# plot fluorescence signals
ax[0].plot(t, f_noisy, label = 'Noisy')
ax[0].plot(t, f, label = 'Noiseless')
legend = ax[0].legend(loc='best', bbox_to_anchor=(1,1))

# plot true and estimated spikes
marker, stem, base = ax[1].stem(t_k, 1 + a_k, color = true_col, label = 'True spike train', markerfmt='x', bottom = 1)
plt.setp(marker, 'color', true_col)
plt.setp(stem, 'color', true_col)
plt.setp(base, 'color', true_col)
marker, stem, base = ax[1].stem(tt_k, np.ones(tt_k.shape), color = est_col, label = 'Estimated spike train', markerfmt='x', bottom = 0)
plt.setp(marker, 'color', est_col)
plt.setp(stem, 'color', est_col)
plt.setp(base, 'color', est_col)
legend = ax[1].legend(loc='best', bbox_to_anchor=(1,1))

# plot pulse trains
ax[2].plot(t_y, y, color = true_col, label = 'True pulse train')
ax[2].plot(t_y, y_hat, color = est_col, label = 'Estimated pulse train', linestyle = ':')
legend = ax[2].legend(loc='best', bbox_to_anchor=(1,1))




