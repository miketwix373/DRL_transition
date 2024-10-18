#!/bin/env python
import numpy as np
import os
import time
import random
from mpi4py import MPI
from PIL import Image

import gymnasium as gym
from gymnasium import spaces
from parameters import params, params_reward
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

from scipy.ndimage import zoom

import tracemalloc

class dynStw(gym.Env):
    # check if this is enough, but it could be not essential
    metadata = {"render_modes": ["console"]}
    
    def __init__(self, render_mode="console"):
        self.gridSize = params["dimGridSize"]
        
        # Beta: img=1 means you are using Cnn
        self.img = 1
        if self.img == 1:
          # Define observation space with min and max values and type
          self.observation_space = spaces.Box(
                  low   = 0,
                  high  = 255,
                  shape = (1,params["target_i"],params["target_j"]),
                  dtype = np.uint8
          )
          self.counter = 0
        else:
          # Define observation space with min and max values and type
          self.observation_space = spaces.Box(
                  low   = params["obsMin"],
                  high  = params["obsMax"],
                  shape = (params["dimGridSize"],),
                  dtype = np.float32
          )
        
        # Define the action, floats between 0 and 1 to be rescaled
        self.action_space = spaces.Box(
            low   = params["actMin"],
            high  = params["actMax"],
            shape = (1,),
            dtype = np.float32
        )
        # Initialise MPI CaNS environment
        self.sub_comm = MPI.COMM_SELF.Spawn('./cans', args=[], \
                        maxprocs=params["maxprocs"][0])
        self.common_comm=self.sub_comm.Merge(False)

        # DEBUG check memory usage
        tracemalloc.start()

        # Define target ratio when performing 2D convolution to undersample image
        self.zoom_i = params["target_i"]/params["size_mat_i"]
        self.zoom_j = params["target_j"]/params["size_mat_j"]

    def get_info(self):
        return 1.
    
    ## THIS IS MANDATORY          
    def reset(self,seed=None):
        info = {}
        
        # Create the file to count the actuation number and store values
        self.f_act = open('count_act.dat','w')
        self.f_act.write(str(0))
        self.f_act.close()

        # For IMG adaptation to CNN
        if self.counter == 0:
          u_obs_all = np.zeros((1,params["target_i"],params["target_j"]),dtype=np.uint8)
          #self.observation = Image.fromarray(u_obs_all,'L') #how about RGB?
          self.observation = u_obs_all
        else:
          u_obs_all = Image.open('./snapshots/img_'+str(self.counter-1)+'.png')
          #self.observation = u_obs_all.convert('L')
          self.img_loaded = np.asarray(u_obs_all)
          self.observation = np.zeros((1,params["target_i"],params["target_j"]),dtype=np.uint8)
          self.observation[0,:,:] = self.img_loaded
          
        print("\nReset done")
        return self.observation, {} 
    
    ## THIS IS MANDATORY
    def step(self,action):
        print("Step called")
       
        # Read the actuation number from a file
        self.f_act   = open('count_act.dat','r')
        self.old_act = self.f_act.read()
        self.f_act.close()

        # Send request to start actuation to CaNS
        if self.old_act=='0':
          req = b'START'
          self.common_comm.Bcast([req, MPI.CHAR], root=0)        
        else:
          req = b'CONTN'
          self.common_comm.Bcast([req, MPI.CHAR], root=0)        

        # Rescale omega and send to CaNS to start the actuation
        self.actions = action
        omega = self.rescale_omega(action)
        self.f   = open('omega_hist.dat','a')
        self.f.write(str(omega))
        self.f.close()
        omega_send = np.array(0,dtype=np.double)
        omega_send = np.double(omega)
        self.f   = open('omega_sent.dat','a')
        self.f.write(str(omega_send))
        self.f.write(str(type(omega_send)))
        self.f.close()
        self.common_comm.Bcast([omega_send, MPI.DOUBLE], root=0)        

        # Wait to receive the observation field and the reward
        u_obs_all = np.zeros(params["size_mat_i"]*params["size_mat_j"])
        dpdx = np.array(0,dtype=np.double)
        e_ks = np.array(0,dtype=np.double)
        self.common_comm.Recv([u_obs_all,      MPI.F_FLOAT],source=1,tag=3)
        self.common_comm.Recv([dpdx,           MPI.DOUBLE] ,source=1,tag=2)
        self.common_comm.Recv([e_ks,           MPI.DOUBLE] ,source=1,tag=1)

        # Send observation as an image for CNN
        obs_mat_big = u_obs_all.reshape((params["size_mat_j"],params["size_mat_i"])).T
        obs_mat = zoom(obs_mat_big,(self.zoom_i, self.zoom_j))
        self.obs_mat = self.img_rescale(obs_mat)
        if int(self.old_act)==params["n_act"]-1:
          self.img_tosave = Image.fromarray(self.obs_mat,'L')
          test = self.img_tosave.save('./snapshots/img_'+str(self.counter)+'.png')
        self.observation = np.zeros((1,params["target_i"],params["target_j"]),dtype=np.uint8)
        self.observation[0,:,:] = self.obs_mat
        self.counter = self.counter +1

        # Compute the reward based on dpdx
        #self.reward = self.comp_reward(dpdx)
        self.reward = self.comp_reward(e_ks)
        self.f   = open('omega_hist.dat','a')
        self.f.write('\n')
        self.f.write(str(omega))
        self.f.close()
        self.f   = open('eks_hist.dat','a')
        self.f.write('\n')
        self.f.write(str(e_ks))
        self.f.close()
        
        # Define the terminate flag
        self.new_act = round(int(self.old_act))+1
        self.f_act   = open('count_act.dat','w')
        self.f_act.write(str(self.new_act))
        self.f_act.close()
        
        print ("i vs params[n_act]-1 = ", self.new_act, "vs", params["n_act"])
        if self.new_act==params["n_act"]:
            self.terminated = True
            self.truncated  = False
            if self.counter == params["n_act"]+params["total_timesteps"]*params["repetitions"]:
              req = b'ENDED'
              # free the (merged) intra communicator
              self.common_comm.Free()
              # disconnect the inter communicator is required to finalize the spawned process.
              self.sub_comm.Disconnect()
            else:
              req = b'CONTN'
            self.common_comm.Bcast([req, MPI.CHAR], root=0)        
            # displaying the memory
            self.f   = open('mem_log.dat','a')
            self.f.write('\n')
            self.f.write(str(tracemalloc.get_traced_memory()))
            self.f.close()
        else:
            self.terminated = False
            self.truncated  = False
            req = b'CONTN'
            self.common_comm.Bcast([req, MPI.CHAR], root=0)        
            
        info = {}
        
        return self.observation, self.reward, self.terminated, self.truncated, {}
    
    def comp_reward(self, dpdx):
        # Normalize the reward to get a float between 0 and 1
        norm_factor = params_reward["dpdx_max"]-params_reward["dpdx_min"]
        self.reward = (dpdx-params_reward["dpdx_min"])/norm_factor
        return self.reward

    def rescale_omega(self, omega):
        # Rescale omega to get a number between omega_min and omega_max
        norm_factor = params["actMax"]-params["actMin"]
        self.omega_rescaled = (omega-params["actMin"])*(params["omMax"]-params["omMin"])/norm_factor \
                              + params["omMin"]
        return self.omega_rescaled

    def img_rescale(self, mat):
        # Rescale array between min and max values given, then to 0 and 255 to get an image
        # Next two lines needed only when limits are imposed
        mat[mat<params["obsMinExp"]] = params["obsMinExp"]
        mat[mat>params["obsMaxExp"]] = params["obsMaxExp"]
        self.res = (mat - params["obsMinExp"])/(params["obsMaxExp"]-params["obsMinExp"]) * 255
        self.res = self.res.astype(np.uint8)
        return self.res

    def reshape_mpi_arr(self, size_box, size_mat, arr):
        # Takes as input an ni*nj 1D array and gives back an (ni,nj) matrix, divided into (I,J) boxes
        # with size (i,j) = (ni/I,nj/J)

        if len(size_box) != 2:
            print ('Error, the box size must be an array with two values')
            return

        if len(size_mat) != 2:
            print ('Error, the mat size must be an array with two values')
            return

        size_mat = np.array(size_mat,np.int64)
        size_box = np.array(size_box,np.int64)
        self.mat = np.zeros(size_mat)
        blocks = size_mat/size_box
        for k in range(1,int(self.multip(blocks))+1):
            row = int((k-1)%blocks[0])
            col = int(np.ceil(k/blocks[0]))-1
            ii  = int(row*size_box[0])
            jj  = int(col*size_box[1])
            temp  = np.reshape(arr[(k-1)*size_box[0]*size_box[1]:\
               (k-1)*size_box[0]*size_box[1] + size_box[0]*size_box[1]],list(size_box))
            self.mat[ii:ii+size_box[0],jj:jj+size_box[1]] = temp

        return self.mat

    def multip(arr):
        res = arr[0]
        for x in arr.flatten():
            res = res*x
        res = res/a[0]
        return res.astype(type(arr[0]))   

