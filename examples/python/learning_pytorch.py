import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.cuda.amp import GradScaler, autocast
from tqdm import trange

import vizdoom as vzd

# Q-learning settings
base_lr = 1.1679909049322575e-05
max_lr = 0.003956458377296167
batch_size = 94
hidden_size = 385
num_layers = 2
n_epochs = 13
gae_lambda = 0.9315539776097506
policy_clip = 0.19695780904032728


cycle_length = 1000
discount_factor = 0.99
train_epochs = 20
learning_steps_per_epoch = 5000

# NN learning settings
batch_size = 94

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (90, 120)  # (height, width)
episodes_to_watch = 10
num_frames = 4

model_savefile = "./model-doom-ppo-cnn-lstm.pth"
save_model = True
load_model = False
skip_learning = False

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.hiddens = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return (np.array(self.states), np.array(self.actions), np.array(self.probs),
                np.array(self.vals), np.array(self.rewards), np.array(self.dones),
                self.hiddens, batches)

    def store_memory(self, state, action, probs, vals, reward, done, hidden):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.hiddens.append(hidden)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.hiddens = []

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
        return np.array(self.frames)

def preprocess(img, resolution):
    img_resized = cv2.resize(img, (resolution[1], resolution[0]), interpolation=cv2.INTER_AREA)
    img_resized = img_resized.astype(np.float32) / 255.0
    return img_resized

class CNNLSTMBase(nn.Module):
    def __init__(self, input_dims, hidden_size=512, num_layers=1):
        super(CNNLSTMBase, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, 8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_size = self._get_cnn_output(input_dims)
        
        self.lstm = nn.LSTM(cnn_out_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        cnn_out = self.cnn(x)
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        lstm_out, hidden = self.lstm(cnn_out.unsqueeze(1), hidden)
        return lstm_out.squeeze(1), hidden

    def _get_cnn_output(self, shape):
        o = self.cnn(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        return (h_0, c_0)

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_size=512):
        super(ActorNetwork, self).__init__()
        self.base = CNNLSTMBase(input_dims, hidden_size)
        self.actor = nn.Linear(hidden_size, n_actions)

    def forward(self, state, hidden=None):
        x, new_hidden = self.base(state, hidden)
        action_probs = F.softmax(self.actor(x), dim=-1)
        return action_probs, new_hidden

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, hidden_size=512):
        super(CriticNetwork, self).__init__()
        self.base = CNNLSTMBase(input_dims, hidden_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state, hidden=None):
        x, new_hidden = self.base(state, hidden)
        value = self.critic(x)
        return value, new_hidden

class CyclicalLR:
    def __init__(self, optimizer, base_lr, max_lr, step_size):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.cycle = 0
        self.step_in_cycle = 0

    def step(self):
        cycle = np.floor(1 + self.step_in_cycle / (2 * self.step_size))
        x = np.abs(self.step_in_cycle / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.step_in_cycle += 1
        if self.step_in_cycle >= 2 * self.step_size:
            self.step_in_cycle = 0
            self.cycle += 1

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class PPOAgent:
    def __init__(self, input_dims, n_actions, num_frames=4, base_lr=0.0003, max_lr=0.001,
                 cycle_length=1000, gamma=0.99, gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=10, entropy_coefficient=0.01, max_grad_norm=0.5,
                 weight_decay=0.01, accumulation_steps=1, n_steps=5):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.n_steps = n_steps

        self.actor = ActorNetwork(input_dims, n_actions).to(DEVICE)
        self.critic = CriticNetwork(input_dims).to(DEVICE)
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), 
                                    lr=base_lr, weight_decay=weight_decay)
        self.scheduler = CyclicalLR(self.optimizer, base_lr, max_lr, cycle_length)

        self.memory = PPOMemory(batch_size)
        self.scaler = GradScaler()
        self.frame_stack = FrameStack(num_frames)

    def store_transition(self, state, action, probs, vals, reward, done, hidden):
        self.memory.store_memory(state, action, probs, vals, reward, done, hidden)

    def choose_action(self, observation, hidden=None):
        state = torch.from_numpy(observation).float().unsqueeze(0).to(DEVICE)
        if hidden is None:
            hidden = self.actor.base.init_hidden(state.size(0))
        
        # Unpack the nested tuple
        hidden_actor, hidden_critic = hidden
        
        with torch.no_grad():
            probs, new_hidden_actor = self.actor(state, hidden_actor)
            value, new_hidden_critic = self.critic(state, hidden_critic)
        
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item(), (new_hidden_actor, new_hidden_critic)

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, hiddens_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(DEVICE)

            values = torch.tensor(values).to(DEVICE)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(DEVICE)
                old_probs = torch.tensor(old_prob_arr[batch]).to(DEVICE)
                actions = torch.tensor(action_arr[batch]).to(DEVICE)
                hidden = (hiddens_arr[batch[0]][0], hiddens_arr[batch[0]][1])

                with autocast():
                    probs, _ = self.actor(states, hidden)
                    dist = Categorical(probs)
                    new_probs = dist.log_prob(actions)
                    prob_ratio = (new_probs - old_probs).exp()
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                                                         1+self.policy_clip) * advantage[batch]
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    critic_value, _ = self.critic(states, hidden)
                    critic_value = critic_value.squeeze()
                    returns = advantage[batch] + values[batch]
                    critic_loss = F.mse_loss(returns, critic_value)

                    entropy = dist.entropy().mean()
                    total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropy
                    total_loss = total_loss / self.accumulation_steps

                self.scaler.scale(total_loss).backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            self.scheduler.step()

        self.memory.clear_memory()

def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

def test(game, agent):
    print("\nTesting...")
    test_scores = []
    for _ in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        agent.frame_stack.reset()
        hidden = agent.actor.base.init_hidden(1)  # Initialize hidden state
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            agent.frame_stack.push(state)
            stacked_state = agent.frame_stack.get()
            action, _, _, new_hidden = agent.choose_action(stacked_state, hidden)
            game.make_action(actions[action], frame_repeat)
            hidden = new_hidden  # Update hidden state
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )

def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    start_time = time()
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        print(f"\nEpoch #{epoch + 1}")
        agent.frame_stack.reset()
        hidden_actor = agent.actor.base.init_hidden(1)
        hidden_critic = agent.critic.base.init_hidden(1)
        hidden = (hidden_actor, hidden_critic)
        
        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer, resolution)
            agent.frame_stack.push(state)
            stacked_state = agent.frame_stack.get()
            action, log_prob, value, new_hidden = agent.choose_action(stacked_state, hidden)
            reward = game.make_action(actions[action], frame_repeat)
            
            done = game.is_episode_finished()

            agent.store_transition(stacked_state, action, log_prob, value, reward, done, new_hidden)

            hidden = new_hidden  # Update hidden state

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()
                agent.frame_stack.reset()
                hidden_actor = agent.actor.base.init_hidden(1)
                hidden_critic = agent.critic.base.init_hidden(1)
                hidden = (hidden_actor, hidden_critic)

        agent.learn()
        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save({
                'actor_state_dict': agent.actor.state_dict()
            }, model_savefile + "_actor.pth")
            torch.save(agent.critic.state_dict(), model_savefile + "_critic.pth")
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game

if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = PPOAgent(
        input_dims=(4, *resolution),
        n_actions=len(actions),
        base_lr=1.1679909049322575e-05,
        max_lr=0.003956458377296167,
        cycle_length=1000,
        gamma=discount_factor,
        batch_size=batch_size,
        n_epochs=13,
        entropy_coefficient=0.01,
        max_grad_norm=0.5,
        weight_decay=0.01,
        accumulation_steps=4,
        n_steps=5  # Add this line
    )

    if load_model:
        print("Loading model from: ", model_savefile)
        agent.actor.load_state_dict(torch.load(model_savefile + "_actor.pth"))
        agent.critic.load_state_dict(torch.load(model_savefile + "_critic.pth"))

    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        agent.frame_stack.reset()  # Reset frame stack at the start of each episode
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            agent.frame_stack.push(state)  # Push new state to frame stack
            stacked_state = agent.frame_stack.get()  # Get stacked state
            best_action_index, _, _ = agent.choose_action(stacked_state)  # Use stacked state

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)