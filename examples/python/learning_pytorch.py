import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent for ViZDoom.")
    
    # Q-learning settings
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='Learning rate for the optimizer')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_steps_per_epoch', type=int, default=4000, help='Number of learning steps per epoch')
    
    # NN learning settings
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    
    # Training regime
    parser.add_argument('--test_episodes_per_epoch', type=int, default=100, help='Number of test episodes per epoch')
    
    # Other parameters
    parser.add_argument('--frame_repeat', type=int, default=12, help='Frame repeat for actions')
    parser.add_argument('--resolution', type=tuple, default=(90, 120), help='Resolution of the game screen (height, width)')
    parser.add_argument('--episodes_to_watch', type=int, default=10, help='Number of episodes to watch after training')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames to stack')
    
    parser.add_argument('--model_savefile', type=str, default="./model-doom-ppo-residual.pth", help='File path to save the model')
    parser.add_argument('--save_model', type=bool, default=True, help='Whether to save the model after training')
    parser.add_argument('--load_model', type=bool, default=False, help='Whether to load a pre-trained model')
    parser.add_argument('--skip_learning', type=bool, default=False, help='Whether to skip the learning phase')
    
    parser.add_argument('--config_file_path', type=str, default=os.path.join(vzd.scenarios_path, "simpler_basic.cfg"), help='Path to the configuration file')
    
    return parser.parse_args()

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

class DiverseExperienceReplay:
    def __init__(self, capacity, num_actions):
        self.capacity = capacity
        self.num_actions = num_actions
        self.buffer = []
        self.action_counts = np.zeros(num_actions)

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            most_common_action = np.argmax(self.action_counts)
            for i, exp in enumerate(self.buffer):
                if exp[1] == most_common_action:
                    del self.buffer[i]
                    self.action_counts[most_common_action] -= 1
                    break
        
        self.buffer.append(experience)
        self.action_counts[experience[1]] += 1

    def sample(self, batch_size):
        probs = 1 / (self.action_counts + 1)
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]

    def generate_batches(self, batch_size):
        buffer_size = len(self.buffer)
        indices = np.arange(buffer_size)
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in range(0, buffer_size, batch_size)]
        return batches

    def clear_memory(self):
        self.buffer = []
        self.action_counts = np.zeros(self.num_actions)

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
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) if len(img_resized.shape) > 2 else img_resized
            processed_buffers.append(img_gray.astype(np.float32) / 255.0)
        else:
            processed_buffers.append(np.zeros(resolution, dtype=np.float32))
    return processed_buffers

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

class NoisyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NoisyNetwork, self).__init__()
        self.fc1 = NoisyLinear(input_dim, 64)
        self.fc2 = NoisyLinear(64, 64)
        self.fc3 = NoisyLinear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()

class MultiBufferActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(MultiBufferActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0] * input_dims[1], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.fc1 = NoisyLinear(conv_out_size, 512)
        self.fc2 = NoisyLinear(512, n_actions)

    def _get_conv_output(self, shape):
        o = F.relu(self.conv1(torch.zeros(1, shape[0] * shape[1], shape[2], shape[3])))
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

class MultiBufferCriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(MultiBufferCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0] * input_dims[1], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        self.flatten = nn.Flatten()
        
        conv_out_size = self._get_conv_output(input_dims)
        
        self.fc1 = NoisyLinear(conv_out_size, 512)
        self.fc2 = NoisyLinear(512, 1)

    def _get_conv_output(self, shape):
        o = F.relu(self.conv1(torch.zeros(1, shape[0] * shape[1], shape[2], shape[3])))
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

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ICM, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Conv3d(state_dim[0], 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._get_conv_output(state_dim), hidden_dim),
            nn.ReLU()
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output = input
        for layer in [nn.Conv3d(shape[0], 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                      nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                      nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))]:
            output = layer(output)
        return int(np.prod(output.size()))

    def forward(self, state, next_state, action):
        state_feat = self.feature(state)
        next_state_feat = self.feature(next_state)
        
        # Inverse Model
        inv_model_input = torch.cat([state_feat, next_state_feat], dim=1)
        pred_action = self.inverse_model(inv_model_input)
        
        # Forward Model
        forward_model_in_features = self.forward_model[0].in_features
        
        num_classes = forward_model_in_features - state_feat.size(1)
        
        action_one_hot = F.one_hot(action, num_classes=num_classes).float()
        action_one_hot = action_one_hot.unsqueeze(0).repeat(state_feat.size(0), 1)  # Ensure batch size matches
        forward_model_input = torch.cat([state_feat, action_one_hot], dim=1)
        
        pred_next_state_feat = self.forward_model(forward_model_input)
        
        return pred_action, pred_next_state_feat, next_state_feat

class MultiBufferPPOAgent:
    def __init__(self, input_dims, n_actions, resolution, base_lr, max_lr, cycle_length,
                 gamma=0.99, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 entropy_coefficient=0.01, max_grad_norm=0.5, weight_decay=0.01, 
                 accumulation_steps=4, n_steps=5):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.entropy_coefficient = entropy_coefficient
        self.n_steps = n_steps
        self.batch_size = batch_size  # Ensure batch_size is assigned to the instance

        self.actor = MultiBufferActorNetwork(input_dims, n_actions).to(DEVICE)
        self.critic = MultiBufferCriticNetwork(input_dims).to(DEVICE)
        
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=base_lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=base_lr, weight_decay=weight_decay)
        
        self.actor_scheduler = CyclicalLR(self.actor_optimizer, base_lr, max_lr, cycle_length)
        self.critic_scheduler = CyclicalLR(self.critic_optimizer, base_lr, max_lr, cycle_length)

        self.memory = DiverseExperienceReplay(capacity=10000, num_actions=n_actions)
        self.frame_stack = MultiBufferFrameStack(input_dims[1], resolution, input_dims[0])
        
        self.scaler = GradScaler()
        
        self.n_actions = n_actions  # Add this line
        state_dim = np.prod(input_dims)  # Calculate total state dimension
        self.icm = ICM(state_dim=input_dims, action_dim=n_actions, hidden_dim=256).to(DEVICE)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)

    def store_transition(self, state, next_state, action, probs, vals, reward, done):
        intrinsic_reward = self.compute_intrinsic_reward(state, next_state, action)
        total_reward = reward + intrinsic_reward
        self.memory.add((state, action, probs, vals, total_reward, done))

    def choose_action(self, observation):
        state = torch.from_numpy(observation).float().unsqueeze(0).to(DEVICE)
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        value = self.critic(state)
        return action.item(), action_log_prob.item(), value.item()

    def calculate_n_step_returns(self, rewards, values, dones):
        n_step_returns = np.zeros_like(rewards)
        for t in range(len(rewards)):
            n_step_return = 0
            for k in range(self.n_steps):
                if t + k < len(rewards):
                    n_step_return += (self.gamma ** k) * rewards[t + k]
                else:
                    break
            if t + self.n_steps < len(values):
                n_step_return += (self.gamma ** self.n_steps) * values[t + self.n_steps]
            n_step_returns[t] = n_step_return
        return n_step_returns

    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(DEVICE)
        action = torch.tensor(action, dtype=torch.long).to(DEVICE)
        
        pred_action, pred_next_state_feat, next_state_feat = self.icm(state, next_state, action)
        
        forward_loss = F.mse_loss(pred_next_state_feat, next_state_feat.detach())
        inverse_loss = F.cross_entropy(pred_action, action.unsqueeze(0))
        
        intrinsic_reward = forward_loss.item()
        
        loss = forward_loss + inverse_loss
        self.icm_optimizer.zero_grad()
        loss.backward()
        self.icm_optimizer.step()
        
        return intrinsic_reward

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr = zip(*self.memory.buffer)
            state_arr = np.array(state_arr)
            action_arr = np.array(action_arr)
            old_prob_arr = np.array(old_prob_arr)
            vals_arr = np.array(vals_arr)
            reward_arr = np.array(reward_arr)
            dones_arr = np.array(dones_arr)

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
            batches = self.memory.generate_batches(self.batch_size)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(DEVICE)
                old_probs = torch.tensor(old_prob_arr[batch]).to(DEVICE)
                actions = torch.tensor(action_arr[batch]).to(DEVICE)

                action_logits = self.actor(states)
                dist = Categorical(logits=action_logits)
                critic_value = self.critic(states).squeeze()

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                entropy = dist.entropy().mean()
                total_loss = actor_loss + 0.5*critic_loss - self.entropy_coefficient*entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

def create_multi_buffer_game(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.init()
    print("Doom initialized.")
    return game

def test(game, agent, actions, test_episodes_per_epoch, frame_repeat, resolution):
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        agent.frame_stack.reset()
        while not game.is_episode_finished():
            state = game.get_state()
            screen_buffer = preprocess_multi_buffer([state.screen_buffer], resolution)[0]
            depth_buffer = preprocess_multi_buffer([state.depth_buffer], resolution)[0]
            labels_buffer = preprocess_multi_buffer([state.labels_buffer], resolution)[0]
            
            agent.frame_stack.push([screen_buffer, depth_buffer, labels_buffer])
            stacked_state = agent.frame_stack.get()
            
            action, _, _ = agent.choose_action(stacked_state)
            game.make_action(actions[action], frame_repeat)
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

def process_state(state, resolution):
    if state is None:
        return np.zeros((3, *resolution), dtype=np.float32)
    screen_buffer = preprocess_multi_buffer([state.screen_buffer], resolution)[0]
    depth_buffer = preprocess_multi_buffer([state.depth_buffer], resolution)[0]
    labels_buffer = preprocess_multi_buffer([state.labels_buffer], resolution)[0]
    return np.array([screen_buffer, depth_buffer, labels_buffer])

def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch, save_model, model_savefile, test_episodes_per_epoch, resolution):
    start_time = time()
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}")
        agent.frame_stack.reset()
        
        for _ in trange(steps_per_epoch, leave=False):
            state = game.get_state()
            screen_buffer = preprocess_multi_buffer([state.screen_buffer], resolution)[0]
            depth_buffer = preprocess_multi_buffer([state.depth_buffer], resolution)[0]
            labels_buffer = preprocess_multi_buffer([state.labels_buffer], resolution)[0]
            
            agent.frame_stack.push([screen_buffer, depth_buffer, labels_buffer])
            stacked_state = agent.frame_stack.get()

            action, log_prob, val = agent.choose_action(stacked_state)
            reward = game.make_action(actions[action], frame_repeat)

            done = game.is_episode_finished()

            if not done:
                next_state = game.get_state()
            else:
                next_state = None

            next_processed_state = process_state(next_state, resolution)
            agent.frame_stack.push(next_processed_state)
            next_stacked_state = agent.frame_stack.get()

            agent.store_transition(stacked_state, next_stacked_state, action, log_prob, val, reward, done)

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()
                agent.frame_stack.reset()

            global_step += 1

        agent.learn()
        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(game, agent, actions, test_episodes_per_epoch, frame_repeat, resolution)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.actor.state_dict(), model_savefile + "_actor.pth")
            torch.save(agent.critic.state_dict(), model_savefile + "_critic.pth")
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game

def main():
    args = parse_args()
    
    # Use the parsed arguments
    learning_rate = args.learning_rate
    discount_factor = args.discount_factor
    train_epochs = args.train_epochs
    learning_steps_per_epoch = args.learning_steps_per_epoch
    batch_size = args.batch_size
    test_episodes_per_epoch = args.test_episodes_per_epoch
    frame_repeat = args.frame_repeat
    resolution = args.resolution
    episodes_to_watch = args.episodes_to_watch
    num_frames = args.num_frames
    model_savefile = args.model_savefile
    save_model = args.save_model
    load_model = args.load_model
    skip_learning = args.skip_learning
    config_file_path = args.config_file_path

    # Initialize game and actions
    game = create_multi_buffer_game(config_file_path)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = MultiBufferPPOAgent(
        input_dims=(3, args.num_frames, *args.resolution),  # 3 buffers: screen, depth, labels
        n_actions=len(actions),
        resolution=args.resolution,
        base_lr=args.learning_rate,
        max_lr=0.001,
        cycle_length=1000,
        gamma=args.discount_factor,
        batch_size=args.batch_size,
        n_epochs=10,
        entropy_coefficient=0.01,
        max_grad_norm=0.5,
        weight_decay=0.01,
        accumulation_steps=4,
        n_steps=5
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
            save_model=save_model,
            model_savefile=model_savefile,
            test_episodes_per_epoch=test_episodes_per_epoch,
            resolution=resolution
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
        agent.frame_stack.reset()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            agent.frame_stack.push(state)
            stacked_state = agent.frame_stack.get()
            best_action_index, _, _ = agent.choose_action(stacked_state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

if __name__ == "__main__":
    main()