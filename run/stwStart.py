import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.ppo.policies import ActorCriticCnnPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stwEnv import dynStw
#from stwEnv import dynStw.common_comm as bla
# but it has to be in stwEnv before class()
from parameters import params, params_reward
from mpi4py import MPI

env = dynStw()
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
#check_env(env)

action_noise =  NormalActionNoise(mean=np.zeros(1), 
                                  sigma=0.1*np.ones(1))
policy_kwargs = dict(
            activation_fn=th.nn.Tanh)

model = PPO("CnnPolicy", 
            env, 
            learning_rate=0.00005, 
            n_steps=params["n_act"], 
            batch_size=8, 
            n_epochs=64, 
            gamma=0.8, 
            gae_lambda=0.95, 
            clip_range=0.2, 
            clip_range_vf=None, 
            normalize_advantage=True, 
            ent_coef=0.1, 
            vf_coef=0.01, 
            max_grad_norm=0.2, 
            use_sde=False, 
            sde_sample_freq=-1, 
            target_kl=None, 
            stats_window_size=1, 
            tensorboard_log="./tensorboardLogs/", 
            policy_kwargs=policy_kwargs, 
            verbose=2, 
            seed=None, 
            device='auto', 
            _init_setup_model=True
            )

print("Model and environment defined")

checkpoint_callback = CheckpointCallback(
                        save_freq=8,
                        save_path="./logs/",
                        name_prefix="PPO"
                        )

model.learn(total_timesteps=params["total_timesteps"],
            callback=None,
            log_interval=4,
            tb_log_name='PPO',
            reset_num_timesteps=True,
            progress_bar=False
            )

model.save('./saved_models')


for x in range(params["repetitions"]):
  custom_obj = {'learning_rate' : 0.00008, 
                'gamma'         : 0.8}
  model = PPO.load('./saved_models',custom_objects=custom_obj)
  #env = model.get_env()
  model.set_env(env)
  model.learn(total_timesteps=params["total_timesteps"],
            callback=None,
            log_interval=4,
            tb_log_name='PPO',
            reset_num_timesteps=False,
            progress_bar=False
            )

  model.save('./saved_models')

  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)

  print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Enjoy trained agent
vec_env = model.get_env()
obs     = vec_env.reset()
for i in range(params["n_act"]):
    action, _states = model.predict(obs, deterministic=True)
    print("Evaluate deterministic policy")
    obs, rewards, terminated, truncated = vec_env.step(action)
    print("Rewards obstained\n")
    print(rewards)

env.close()
