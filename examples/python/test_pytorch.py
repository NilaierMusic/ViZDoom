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
config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")

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
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(input_dims[0] * input_dims[3], 32, 8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.res3 = ResidualBlock(64)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def _get_conv_output(self, shape):
        # shape is (num_frames, height, width, channels)
        # we need to create a tensor of shape (1, num_frames * channels, height, width)
        o = torch.zeros(1, shape[0] * shape[3], shape[1], shape[2])
        o = F.relu(self.bn1(self.conv1(o)))
        o = self.res1(o)
        o = F.relu(self.bn2(self.conv2(o)))
        o = self.res2(o)
        o = F.relu(self.bn3(self.conv3(o)))
        o = self.res3(o)
        return int(np.prod(o.size()))

    def forward(self, state, hidden=None):
        batch_size = state.size(0)
        # Reshape the input: [batch_size, num_frames, height, width, channels] to [batch_size, num_frames * channels, height, width]
        x = state.view(batch_size, self.input_dims[0] * self.input_dims[3], self.input_dims[1], self.input_dims[2])
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res3(x)
        x = self.flatten(x)
        x = x.unsqueeze(1)  # Add time dimension
        
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs, hidden

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(8, 32, 8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.res3 = ResidualBlock(64)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        self.fc1 = nn.Linear(512, 256)  # Changed back to fc1 and fc2
        self.fc2 = nn.Linear(256, 1)

    def _get_conv_output(self, shape):
        # Use 8 channels instead of shape[0] * shape[3]
        o = torch.zeros(1, 8, shape[1], shape[2])
        o = F.relu(self.bn1(self.conv1(o)))
        o = self.res1(o)
        o = F.relu(self.bn2(self.conv2(o)))
        o = self.res2(o)
        o = F.relu(self.bn3(self.conv3(o)))
        o = self.res3(o)
        return int(np.prod(o.size()))

    def forward(self, state, hidden=None):
        batch_size = state.size(0)
        # Reshape the input to have 8 channels
        x = state.view(batch_size, 8, self.input_dims[1], self.input_dims[2])
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res3(x)
        x = self.flatten(x)
        x = x.unsqueeze(1)  # Add time dimension
        
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value, hidden

class FrameStack:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

    def reset(self):
        for _ in range(self.num_frames):
            self.frames.append(np.zeros((*resolution, 2), dtype=np.float32))

    def push(self, frame):
        self.frames.append(frame)

    def get(self):
        # Stack frames along the channel dimension
        return np.concatenate(self.frames, axis=-1)

# Converts and down-samples the input image
def preprocess(img, resolution):
    img_resized = cv2.resize(img, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.float32) / 255.0
    # Add a second channel (you might want to use a different processing for the second channel)
    img_2channel = np.stack([img_resized, img_resized], axis=-1)
    return img_2channel

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

    # Create instances of the actor and critic networks
    actor = ActorNetwork((4, *resolution, 2), len(actions)).to(DEVICE)
    critic = CriticNetwork((4, *resolution, 2)).to(DEVICE)  # Change to 2 channels

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
        frame_stack.reset()  # Reset the frame stack at the start of each episode
        hidden_actor = None
        hidden_critic = None
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            frame_stack.push(state)
            stacked_state = frame_stack.get()
            
            stacked_state = torch.from_numpy(stacked_state).float().unsqueeze(0).to(DEVICE)
            
            probs, hidden_actor = actor(stacked_state, hidden_actor)
            _, best_action_index = torch.max(probs, dim=1)
            best_action_index = best_action_index.item()

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()
                sleep(0.03)

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)