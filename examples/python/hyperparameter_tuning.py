# hyperparameter_tuning.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import vizdoom as vzd
import cv2
import numpy as np
import optuna
from tqdm import trange
from collections import deque

# Global parameters
frame_repeat = 4
resolution = (90, 120)
num_frames = 4
test_episodes_per_epoch = 10
config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")
actions = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]  # Example action space

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        batch_size = x.size(0)
        cnn_out = self.cnn(x)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        lstm_out, new_hidden = self.lstm(cnn_out.unsqueeze(1), hidden)
        return lstm_out.squeeze(1), new_hidden

    def _get_cnn_output(self, shape):
        o = self.cnn(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        return (h_0, c_0)

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_size=512, num_layers=1):
        super(ActorNetwork, self).__init__()
        self.base = CNNLSTMBase(input_dims, hidden_size, num_layers)
        self.actor = nn.Linear(hidden_size, n_actions)

    def forward(self, state, hidden=None):
        x, new_hidden = self.base(state, hidden)
        action_probs = nn.functional.softmax(self.actor(x), dim=-1)
        return action_probs, new_hidden

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, hidden_size=512, num_layers=1):
        super(CriticNetwork, self).__init__()
        self.base = CNNLSTMBase(input_dims, hidden_size, num_layers)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state, hidden=None):
        x, new_hidden = self.base(state, hidden)
        value = self.critic(x)
        return value, new_hidden

class PPOAgent:
    def __init__(self, input_dims, n_actions, num_frames=4, base_lr=0.0003, max_lr=0.001,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10, entropy_coefficient=0.01,
                 hidden_size=512, num_layers=1):
        self.gamma = 0.99
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient

        self.actor = ActorNetwork(input_dims, n_actions, hidden_size, num_layers).to(DEVICE)
        self.critic = CriticNetwork(input_dims, hidden_size, num_layers).to(DEVICE)

        parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(parameters, lr=base_lr)

        self.memory = PPOMemory(batch_size)
        self.frame_stack = FrameStack(num_frames)

    def store_transition(self, state, action, probs, vals, reward, done, hidden):
        self.memory.store_memory(state, action, probs, vals, reward, done, hidden)

    def choose_action(self, observation, hidden=None):
        state = torch.from_numpy(observation).float().unsqueeze(0).to(DEVICE)
        if hidden is None:
            hidden_actor = self.actor.base.init_hidden(1)
            hidden_critic = self.critic.base.init_hidden(1)
        else:
            hidden_actor, hidden_critic = hidden

        with torch.no_grad():
            probs, new_hidden_actor = self.actor(state, hidden_actor)
            value, new_hidden_critic = self.critic(state, hidden_critic)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item(), (new_hidden_actor, new_hidden_critic)

    def learn(self):
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, hiddens_arr, batches = \
            self.memory.generate_batches()

        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = torch.tensor(advantage).to(DEVICE)

        values = torch.tensor(values).to(DEVICE)
        for batch in batches:
            states = torch.tensor(state_arr[batch], dtype=torch.float).to(DEVICE)
            old_probs = torch.tensor(old_prob_arr[batch]).to(DEVICE)
            actions = torch.tensor(action_arr[batch]).to(DEVICE)
            hidden = (hiddens_arr[batch[0]][0], hiddens_arr[batch[0]][1])

            probs, _ = self.actor(states, hidden)
            dist = Categorical(probs)
            new_probs = dist.log_prob(actions)
            prob_ratio = (new_probs - old_probs).exp()
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

            critic_value, _ = self.critic(states, hidden)
            critic_value = critic_value.squeeze()
            returns = advantage[batch] + values[batch]
            critic_loss = nn.functional.mse_loss(returns, critic_value)

            entropy = dist.entropy().mean()
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.memory.clear_memory()

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
        batches = [indices[i:i + self.batch_size] for i in batch_start]
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

def create_simple_game():
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    return game

def evaluate_agent(trial):
    base_lr = trial.suggest_float("base_lr", 1e-5, 1e-3, log=True)
    max_lr = trial.suggest_float("max_lr", 1e-4, 1e-2, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    policy_clip = trial.suggest_float("policy_clip", 0.1, 0.3)
    batch_size = trial.suggest_int("batch_size", 32, 128)
    hidden_size = trial.suggest_int("hidden_size", 128, 1024)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    n_epochs = trial.suggest_int("n_epochs", 3, 20)

    game = create_simple_game()
    agent = PPOAgent((num_frames, resolution[0], resolution[1]), len(actions),
                     num_frames=num_frames, base_lr=base_lr, max_lr=max_lr, gae_lambda=gae_lambda,
                     policy_clip=policy_clip, batch_size=batch_size, n_epochs=n_epochs,
                     hidden_size=hidden_size, num_layers=num_layers)

    print("\nEvaluating...")
    test_scores = []
    for _ in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        agent.frame_stack.reset()
        hidden = None
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            agent.frame_stack.push(state)
            stacked_state = agent.frame_stack.get()
            action, _, _, new_hidden = agent.choose_action(stacked_state, hidden)
            game.make_action(actions[action], frame_repeat)
            hidden = new_hidden
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    mean_score = np.mean(test_scores)
    game.close()
    return mean_score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(evaluate_agent, n_trials=50)
    print("Best hyperparameters:", study.best_params)