#!/bin/env python
import numpy as np

# All the possible parameters for the simulation that DRL needs
# are listed below

# Size fo the observation grid (nx,ny,n_variables,n_snapshots)
obsGridParams    = [48,32,1,1]
for i in range(len(obsGridParams)):
    if i==0:
        obsGridSize = obsGridParams[i]
    else:
        obsGridSize = obsGridSize*obsGridParams[i]

# Number of actions to be given as output by DRL.
# They correspond to the number of actuations performed in an episode
n_actions = 16

# Number of actuations per episode
n_actuations = n_actions

# Minimum and maximum wanted values in the observation grid
obsMin = -1.0
obsMax = 1.0

# Minimum and maximum expected values in the observation grid
obsMinExp = 0.65
obsMaxExp = 1.3

# Minimum and maximum values of the *unshifted* action
actMin = -1.0
actMax = 1.0

# Minimum and maximum values of the *shifted* action
omMin = -0.5
omMax = 1.

# Total number of timesteps during the training
total_timesteps = n_actions*400

# Total number of processes in canscomm
maxprocs = np.arange(1, dtype='int32')
maxprocs[0]=32

# Grid params for CNN
size_box_i = 4
size_box_j = 12
size_mat_i = 32
size_mat_j = 48

# The following values are passed to the environment
params = {
    "n_act"           : n_actuations,
    "n_a"             : n_actions,
    "dimGridParams"   : obsGridParams,
    "dimGridSize"     : obsGridSize,
    "obsMin"          : obsMin,
    "obsMax"          : obsMax,
    "obsMinExp"       : obsMinExp,
    "obsMaxExp"       : obsMaxExp,
    "actMin"          : actMin,
    "actMax"          : actMax,
    "omMin"           : omMin,
    "omMax"           : omMax,
    "total_timesteps" : total_timesteps,
    "maxprocs"        : maxprocs,
    "size_box_i"      : size_box_i,
    "size_box_j"      : size_box_j,
    "size_mat_i"      : size_mat_i,
    "size_mat_j"      : size_mat_j,
}

### Reward related

params_reward = {
    "reward_function": "dpdx",
    "dpdx_min": 50,
    "dpdx_max": 150,
}
