# %%
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# %%
# done = True
# # loop through frames of the game
# for frame in range(10000):
#     if done:
#         # start game
#         env.reset()
#     # do random action
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()
# env.close()
# %%
''' 
prepocessing 
1. Create base environment
2. Simpilfy controls
3. grayscale 
4. Warp in dummy environment
4. frame stacking
'''
from gym.wrappers import GrayScaleObservation
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# game environment setup
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# limiting actions (simplification)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# grayscaling
env = GrayScaleObservation(env, keep_dim=True)
# wrapping in dummy environemnt
env = DummyVecEnv([lambda: env])
# stacking
num_frames = 4
env = VecFrameStack(env, num_frames, channels_order='last')
# %%
'''
Training the reinforment learning model
using  Proximal Policy Optimization
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
'''
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
model =  PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 
model.learn(total_timesteps=1000000, callback=callback)
# model.save('thisisatestmodel')
# %%
 