#!/usr/bin/env python3

import os
import itertools as it
from time import sleep

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import vizdoom as vzd

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

# Testing settings
episodes_to_watch = 10
frame_repeat = 12

resolution = (90, 120)  # (height, width)

# Model saving and loading parameters
model_savefile = "./model-doom-ppo-residual"
actor_model_savefile = model_savefile + ".pth_actor.pth"
critic_model_savefile = model_savefile + ".pth_critic.pth"

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0] * input_dims[1], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.fc1 = NoisyLinear(conv_out_size, 512)
        self.fc2 = NoisyLinear(512, n_actions)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, shape[0] * shape[1], shape[2], shape[3])
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, state):
        x = state.view(state.size(0), -1, state.size(-2), state.size(-1))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        action_logits = self.fc2(x)
        return action_logits

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0] * input_dims[1], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.fc1 = NoisyLinear(conv_out_size, 512)
        self.fc2 = NoisyLinear(512, 1)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, shape[0] * shape[1], shape[2], shape[3])
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, state):
        x = state.view(state.size(0), -1, state.size(-2), state.size(-1))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

class MultiBufferFrameStack:
    def __init__(self, num_frames, resolution, num_buffers):
        self.num_frames = num_frames
        self.resolution = resolution
        self.num_buffers = num_buffers
        self.frames = [deque([], maxlen=num_frames) for _ in range(num_buffers)]

    def reset(self):
        for buffer in self.frames:
            for _ in range(self.num_frames):
                buffer.append(np.zeros(self.resolution, dtype=np.float32))

    def push(self, frames):
        for i, frame in enumerate(frames):
            self.frames[i].append(frame)

    def get(self):
        return np.array([np.array(buffer) for buffer in self.frames])

def preprocess_multi_buffer(buffers, resolution):
    processed_buffers = []
    for buffer in buffers:
        if buffer is not None:
            img_resized = cv2.resize(buffer, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
            img_resized = img_resized.astype(np.float32) / 255.0
            processed_buffers.append(img_resized)
        else:
            processed_buffers.append(np.zeros(resolution, dtype=np.float32))
    return processed_buffers

# Creates and initializes ViZDoom environment
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_depth_buffer_enabled(True)
    game.init()
    print("Doom initialized.")
    return game

if __name__ == "__main__":
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    n_actions = len(actions)
    print(f"Number of actions: {n_actions}")
    expected_n_actions = 128  # 2^15, based on 15 available buttons
    if n_actions != expected_n_actions:
        # Adjust the number of actions to match the saved model
        actions = actions[:expected_n_actions]
        n_actions = len(actions)
        print(f"Adjusted number of actions to match the saved model: {n_actions}")

    if n_actions != expected_n_actions:
        raise ValueError(f"Number of actions ({n_actions}) does not match the saved model's number of actions ({expected_n_actions}).")

    # Create instances of the actor and critic networks
    actor = ActorNetwork((3, 4, *resolution), expected_n_actions).to(DEVICE)
    critic = CriticNetwork((3, 4, *resolution)).to(DEVICE)

    print("Loading actor model from: ", actor_model_savefile)
    actor.load_state_dict(torch.load(actor_model_savefile, map_location=DEVICE))
    print("Loading critic model from: ", critic_model_savefile)
    critic.load_state_dict(torch.load(critic_model_savefile, map_location=DEVICE))

    print("======================================")
    print("Testing trained neural network!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    frame_stack = MultiBufferFrameStack(4, resolution, 3)  # Create a frame stack with 4 frames and 3 buffers

    for _ in range(episodes_to_watch):
        game.new_episode()
        frame_stack.reset()
        while not game.is_episode_finished():
            state = game.get_state()
            screen_buffer = preprocess_multi_buffer([state.screen_buffer], resolution)[0]
            depth_buffer = preprocess_multi_buffer([state.depth_buffer], resolution)[0]
            labels_buffer = preprocess_multi_buffer([state.labels_buffer], resolution)[0]

            frame_stack.push([screen_buffer, depth_buffer, labels_buffer])
            stacked_state = frame_stack.get()
            
            stacked_state = torch.from_numpy(stacked_state).float().unsqueeze(0).to(DEVICE)
            
            action_logits = actor(stacked_state)
            probs = F.softmax(action_logits, dim=-1)
            _, best_action_index = torch.max(probs, dim=1)
            best_action_index = best_action_index.item()

            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
                sleep(0.03)

        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)