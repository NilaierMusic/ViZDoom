#!/usr/bin/env python3

import os
import itertools as it
from time import sleep

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vizdoom as vzd

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

# Testing settings
episodes_to_watch = 10
frame_repeat = 12

resolution = (90, 120)  # (height, width)

# Model saving and loading parameters
model_savefile = "./model-doom-ppo"
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

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, n_actions)

    def _get_conv_output(self, shape):
        o = self.bn1(self.conv1(torch.zeros(1, *shape)))
        o = self.bn2(self.conv2(o))
        o = self.bn3(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, state, hidden=None):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = x.unsqueeze(1)  # Add time dimension
        
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        
        x = x.squeeze(1)
        action_probs = F.softmax(self.fc(x), dim=-1)
        return action_probs, hidden

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.lstm = nn.LSTM(conv_out_size, 512, batch_first=True)
        self.fc = nn.Linear(512, 1)

    def _get_conv_output(self, shape):
        o = self.bn1(self.conv1(torch.zeros(1, *shape)))
        o = self.bn2(self.conv2(o))
        o = self.bn3(self.conv3(o))
        return int(np.prod(o.size()))

    def forward(self, state, hidden=None):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = x.unsqueeze(1)  # Add time dimension
        
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        
        x = x.squeeze(1)
        value = self.fc(x)
        return value, hidden

# Converts and down-samples the input image
def preprocess(img, resolution):
    # Assuming img is already a numpy array of shape (480, 640)
    # Resize the image to the specified resolution using OpenCV
    img_resized = cv2.resize(img, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    img_resized = img_resized.astype(np.float32) / 255.0
    
    # Reshape to (channel, height, width)
    img_resized = img_resized.reshape(1, *img_resized.shape)
    
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

    # Create instances of the actor and critic networks
    actor = ActorNetwork((1, *resolution), len(actions)).to(DEVICE)
    critic = CriticNetwork((1, *resolution)).to(DEVICE)

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

    for _ in range(episodes_to_watch):
        game.new_episode()
        hidden_actor = None
        hidden_critic = None
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            
            probs, hidden_actor = actor(state, hidden_actor)
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