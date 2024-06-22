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

class PPOMemory:
    def __init__(self, batch_size, n_steps):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.n_steps = n_steps

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

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

class FrameStack:
    def __init__(self, num_frames, resolution):
        self.num_frames = num_frames
        self.resolution = resolution
        self.frames = deque([], maxlen=num_frames)

    def reset(self):
        for _ in range(self.num_frames):
            self.frames.append(np.zeros(self.resolution, dtype=np.float32))

    def push(self, frame):
        self.frames.append(frame)

    def get(self):
        return np.array(self.frames)

def preprocess(img, resolution):
    img_resized = cv2.resize(img, (resolution[1], resolution[0]), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.float32) / 255.0
    return img_resized

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
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4, padding=2)
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
        o = F.relu(self.bn1(self.conv1(torch.zeros(1, *shape))))
        o = self.res1(o)
        o = F.relu(self.bn2(self.conv2(o)))
        o = self.res2(o)
        o = F.relu(self.bn3(self.conv3(o)))
        o = self.res3(o)
        return int(np.prod(o.size()))

    def forward(self, state, hidden=None):
        x = F.relu(self.bn1(self.conv1(state)))
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
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4, padding=2)
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
        self.fc2 = nn.Linear(256, 1)

    def _get_conv_output(self, shape):
        o = F.relu(self.bn1(self.conv1(torch.zeros(1, *shape))))
        o = self.res1(o)
        o = F.relu(self.bn2(self.conv2(o)))
        o = self.res2(o)
        o = F.relu(self.bn3(self.conv3(o)))
        o = self.res3(o)
        return int(np.prod(o.size()))

    def forward(self, state, hidden=None):
        x = F.relu(self.bn1(self.conv1(state)))
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
    def __init__(self, input_dims, n_actions, resolution, num_frames=4, base_lr=0.0001, max_lr=0.001, cycle_length=1000,
                 gamma=0.99, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 entropy_coefficient=0.01, max_grad_norm=0.5, weight_decay=0.01, accumulation_steps=4, n_steps=5):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.num_frames = num_frames
        self.entropy_coefficient = entropy_coefficient  # Ensure this line is present

        self.actor = ActorNetwork(input_dims, n_actions).to(DEVICE)
        self.critic = CriticNetwork(input_dims).to(DEVICE)
        
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=base_lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=base_lr, weight_decay=weight_decay)
        
        self.actor_scheduler = CyclicalLR(self.actor_optimizer, base_lr, max_lr, cycle_length)
        self.critic_scheduler = CyclicalLR(self.critic_optimizer, base_lr, max_lr, cycle_length)

        self.memory = PPOMemory(batch_size, n_steps)
        self.hidden_actor = None
        self.hidden_critic = None
        self.scaler = GradScaler()
        self.total_steps = 0
        self.frame_stack = FrameStack(num_frames, resolution)
        
        self.n_steps = n_steps

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        if observation.ndim == 2:  # Single frame
            observation = np.stack([observation] * self.num_frames)
        state = torch.from_numpy(observation).float().unsqueeze(0).to(DEVICE)
        probs, self.hidden_actor = self.actor(state, self.hidden_actor)
        probs = probs.squeeze(0)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value, self.hidden_critic = self.critic(state, self.hidden_critic)
        return action.item(), log_prob.item(), value.item()

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

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            n_step_returns = self.calculate_n_step_returns(reward_arr, values, dones_arr)
            advantage = n_step_returns - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            values = torch.tensor(values, dtype=torch.float32).to(DEVICE)
            advantage = torch.tensor(advantage, dtype=torch.float32).to(DEVICE)
            n_step_returns = torch.tensor(n_step_returns, dtype=torch.float32).to(DEVICE)
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            for i, batch in enumerate(batches):
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(DEVICE)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float32).to(DEVICE)
                actions = torch.tensor(action_arr[batch], dtype=torch.long).to(DEVICE)

                with autocast():
                    probs, _ = self.actor(states)
                    dist = Categorical(probs)
                    new_probs = dist.log_prob(actions)
                    prob_ratio = (new_probs - old_probs).exp()
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                                                         1+self.policy_clip) * advantage[batch]
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                    
                    critic_value, _ = self.critic(states)
                    critic_value = critic_value.squeeze()
                    returns = n_step_returns[batch]
                    critic_loss = F.mse_loss(returns, critic_value)

                    entropy = dist.entropy().mean()
                    total_loss = (actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropy) / self.accumulation_steps
                
                    self.scaler.scale(total_loss).backward()
                
                if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(batches):
                    self.scaler.unscale_(self.actor_optimizer)
                    self.scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                    
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    self.actor_scheduler.step()
                    self.critic_scheduler.step()

                self.total_steps += 1

        self.memory.clear_memory()
        self.hidden_actor = None
        self.hidden_critic = None

def create_simple_game(config_file_path):
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

def test(game, agent, test_episodes_per_epoch, frame_repeat, resolution):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        agent.frame_stack.reset()  # Reset the frame stack at the start of each episode
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer, resolution)
            agent.frame_stack.push(state)  # Push the new state to the frame stack
            stacked_state = agent.frame_stack.get()  # Get the stacked state
            action, _, _ = agent.choose_action(stacked_state)  # Pass the stacked state to choose_action
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

def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch, resolution, save_model, model_savefile, test_episodes_per_epoch):
    start_time = time()
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}")
        agent.hidden_critic = None
        agent.frame_stack.reset()
        
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_values = []
        episode_rewards = []
        episode_dones = []
        
        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer, resolution)
            agent.frame_stack.push(state)
            stacked_state = agent.frame_stack.get()
            action, log_prob, value = agent.choose_action(stacked_state)
            reward = game.make_action(actions[action], frame_repeat)
            
            if actions[action] == [0, 0, 0]:
                reward -= 0.1
            
            done = game.is_episode_finished()

            episode_states.append(stacked_state)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_values.append(value)
            episode_rewards.append(reward)
            episode_dones.append(done)

            if done or len(episode_states) == agent.n_steps:
                if not done:
                    _, _, last_value = agent.choose_action(stacked_state)
                else:
                    last_value = 0
                
                n_step_returns = agent.calculate_n_step_returns(episode_rewards, episode_values + [last_value], episode_dones)
                
                for i in range(len(episode_states)):
                    agent.store_transition(episode_states[i], episode_actions[i], episode_log_probs[i], 
                                           episode_values[i], n_step_returns[i], episode_dones[i])
                
                episode_states = []
                episode_actions = []
                episode_log_probs = []
                episode_values = []
                episode_rewards = []
                episode_dones = []

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()
                agent.hidden_actor = None
                agent.hidden_critic = None
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

        test(game, agent, test_episodes_per_epoch, frame_repeat, resolution)
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
    game = create_simple_game(config_file_path)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = PPOAgent(
        input_dims=(4, *resolution),
        n_actions=len(actions),
        resolution=resolution,  # Pass the resolution here
        base_lr=learning_rate,
        max_lr=0.001,
        cycle_length=1000,
        gamma=discount_factor,
        batch_size=batch_size,
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
            resolution=resolution,
            save_model=save_model,
            model_savefile=model_savefile,
            test_episodes_per_epoch=test_episodes_per_epoch  # Pass the variable here
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

if __name__ == "__main__":
    main()