import gymnasium as gym
import torch as th
import torch.nn as nn
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
#global common_comm

env = dynStw()
#check_env(env)
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        #features_dimension = 1
        # Check how to compare it with features_dim: int = 256
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

action_noise =  NormalActionNoise(mean=np.zeros(1), 
                                  sigma=0.1*np.ones(1))
#policy_kwargs = dict(
#            activation_fn=th.nn.Tanh,
#            net_arch=[512,512],
#            )
policy_kwargs = dict(
            activation_fn=th.nn.Tanh,
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=32),
            #net_arch=dict(pi=[64, 64], qf=[256, 256]))
            net_arch=dict(pi=[32, 32], qf=[256, 256]))

model = PPO("CnnPolicy", 
            env, 
            learning_rate=0.0001, 
            n_steps=params["n_act"], 
            batch_size=8, 
            n_epochs=32, 
            gamma=0.99, 
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
            tensorboard_log="./log/", 
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
                        name_prefix="DDPG"
                        )


model.learn(total_timesteps=params["total_timesteps"],
            callback=None,
            log_interval=4,
            tb_log_name='DDPG',
            reset_num_timesteps=True,
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

# env.close()
