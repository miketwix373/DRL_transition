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
                  shape = (params["size_mat_i"],params["size_mat_j"]),
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
        #mpi_info = MPI.Info.Create()
        #mpi_info.Set("hostfile", "./mpirun_hosts")
        self.sub_comm = MPI.COMM_SELF.Spawn('./cans', args=[], \
                        maxprocs=params["maxprocs"][0])
        self.common_comm=self.sub_comm.Merge(False)

    def get_info(self):
        return 1.
    
    ## THIS IS MANDATORY          
    def reset(self,seed=None):
        info = {}
        
        # Create the file to count the actuation number and store values
        self.f_act = open('count_act.dat','w')
        self.f_act.write(str(0))
        self.f_act.close()

        # For Multi Input Policy
        #u_obs_all = np.zeros(48*32)
        #self.observation = u_obs_all     

        # For IMG adaptation to CNN
        u_obs_all = np.zeros((params["size_mat_i"],params["size_mat_j"]),dtype=np.uint8)
        self.observation = Image.fromarray(u_obs_all,'L') #how about RGB?
        #self.observation = u_obs_all

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
        u_obs_all = np.zeros(48*32)
        dpdx = np.array(0,dtype=np.double)
        e_ks = np.array(0,dtype=np.double)
        self.common_comm.Recv([u_obs_all,      MPI.F_FLOAT],source=1,tag=3)
        self.common_comm.Recv([dpdx,           MPI.DOUBLE] ,source=1,tag=2)
        self.common_comm.Recv([e_ks,           MPI.DOUBLE] ,source=1,tag=1)
        #self.observation = u_obs_all     
        # Send observation as an image for CNN
        # The procedure below was needed for the old MPI communication
        #obs_mat = self.reshape_mpi_arr(np.array([params["size_box_i"], params["size_box_j"]]),\
        #                               np.array([params["size_mat_i"], params["size_mat_j"]]),\
        #                               u_obs_all)
        obs_mat = u_obs_all.reshape((48,32)).T
        self.obs_mat = self.img_rescale(obs_mat)
        self.observation = Image.fromarray(self.obs_mat,'L')
        test = self.observation.save('./snapshots/img_'+str(self.counter)+'.png')
        self.counter = self.counter +1
        #self.observation = self.obs_mat

        # Compute the reward based on dpdx
        self.reward = self.comp_reward(dpdx)
        self.f   = open('omega_hist.dat','a')
        self.f.write(str(omega))
        self.f.close()
        self.f   = open('dpdx_hist.dat','a')
        self.f.write(str(dpdx))
        self.f.close()
        
        # Define the terminate flag
        self.new_act = round(int(self.old_act))+1
        self.f_act   = open('count_act.dat','w')
        self.f_act.write(str(self.new_act))
        self.f_act.close()
        
        print ("i vs params[n_act]-1 = ", self.new_act, "vs", params["n_act"])
        if self.new_act==params["n_act"]:
            self.f = open('dpdx_hist.dat','a')
            self.f.write('\n')
            self.f.write(str(self.reward))
            self.f.close()
            self.terminated = True
            self.truncated  = False
            req = b'CONTN'
            self.common_comm.Bcast([req, MPI.CHAR], root=0)        
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
        ## This is needed if blocks are not in lexic-numerical order
        #for i in range(int(params["size_mat_i"]/params["size_box_i"])):
        #  if i != 0 and mod(i,2) != 0:
        #    start = int(i*params["size_box_i"])
        #    end   = start + int(params["size_box_i"])-1
        #    self.res[:,start:end] = np.flipud(self.res[:,start:end])
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

    def multip(self,arr):
       res = 1.
       for x in arr.flatten():
         res = res*x

       return res

