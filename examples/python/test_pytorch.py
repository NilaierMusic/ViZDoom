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
config_file_path = os.path.join(vzd.scenarios_path, "doom.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        action_logits = self.fc2(x)
        return action_logits

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 1)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

class FrameStack:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

    def reset(self):
        for _ in range(self.num_frames):
            self.frames.append(np.zeros(resolution, dtype=np.float32))

    def push(self, frame):
        self.frames.append(frame)

    def get(self):
        return np.stack(self.frames, axis=0)

# Converts and down-samples the input image
def preprocess(img, resolution):
    img_resized = cv2.resize(img, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.float32) / 255.0
    return img_resized

# Creates and initializes ViZDoom environment
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

if __name__ == "__main__":
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Ensure the number of actions matches the saved model
    n_actions = len(actions)
    if n_actions != 8:
        raise ValueError(f"Number of actions ({n_actions}) does not match the saved model's number of actions (8).")

    # Create instances of the actor and critic networks
    actor = ActorNetwork((4, *resolution), n_actions).to(DEVICE)
    critic = CriticNetwork((4, *resolution)).to(DEVICE)

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

    frame_stack = FrameStack(4)  # Create a frame stack with 4 frames

    for _ in range(episodes_to_watch):
        game.new_episode()
        frame_stack.reset()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            frame_stack.push(state)
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