import itertools as it
import os
import random
from collections import deque
from time import time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
import imageio
import matplotlib.pyplot as plt
import cv2
from network import define_Transformer

experiment_name = "Deep_Q-learning"
model_savefile = os.path.join("results", experiment_name, "models/model-test.pth")
mode = 'Train' # 'Train' or 'Test'
os.makedirs(os.path.join("results", experiment_name, "models"), exist_ok=True)
os.makedirs(os.path.join("results", experiment_name, "videos"), exist_ok=True)

# Q-learning settings
learning_rate = 1e-4
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 5000

# NN learning settings
batch_size = 128

# Rewards
action_reward = -0.4
killing_reward = 100

# Other parameters
# MOVE_LEFT, MOVE_RIGHT, STAY, MOVE_LEFT + ATTACK, MOVE_RIGHT + ATTACK, ATTACK
actions = [[True, False, False], [False, True, False], [False, False, False], [True, False, True], [False, True, True], [False, False, True]]
resolution = (30, 45) # Downsampled resolution of the input image (480*640)
episodes_test = 5
max_steps = 200

if mode == 'Test':
    save_model = False
    load_model = True
    skip_learning = True
else:
    save_model = True
    load_model = False
    skip_learning = False

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
    game.set_doom_map("map01")
    game.set_available_buttons(
        [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
    )
    game.set_episode_timeout(max_steps*100)
    game.set_episode_start_time(10)
    game.set_living_reward(0)
    # number of kills, health, bullets, hit, death
    game.set_available_game_variables([vzd.GameVariable.KILLCOUNT, vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.HITCOUNT, vzd.GameVariable.DEATHCOUNT])
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game

def get_rewards(variables_current, variables_last=None):
    """
    Returns the reward based on the game variables.
    The variables are in the order: [kills, health, bullets, hit, death]
    """
    kills, health, bullets, hit, death = variables_current
    if variables_last is not None:
        kills_last, health_last, bullets_last, hit_last, death_last = variables_last
        kills = kills - kills_last
    reward = kills * killing_reward + action_reward

    return reward

def calculate_smooth_and_variance(data, window_size):
    """
    Calculates the smoothed data and variance for the given data using a moving window.
    """
    new_data = []
    variances = []
    for i in range(len(data)):
        if i < window_size:
            variances.append(np.sqrt(np.var(data[:i+1])))
            new_data.append(sum(data[:i+1]) / (i+1))
        else:
            variances.append(np.sqrt(np.var(data[i-window_size:i])))
            new_data.append(sum(data[i-window_size:i]) / window_size)
    return new_data, variances

def draw_results(all_results, save_path):
    """
    Draws the training results.
    """
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(12, 6))
    all_results_smooth, all_results_variance = calculate_smooth_and_variance(all_results, 20)
    all_results_upper = [all_results_smooth[i] + all_results_variance[i] for i in range(len(all_results_smooth))]
    all_results_lower = [all_results_smooth[i] - all_results_variance[i] for i in range(len(all_results_smooth))]
    plt.plot(all_results_smooth, color='blue')
    plt.fill_between(
        range(len(all_results_smooth)),
        all_results_lower,
        all_results_upper,
        color='blue',
        alpha=0.2,
    )
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training Results')
    plt.grid()
    plt.savefig(save_path, dpi=300)
    plt.close('all')

def run(game, agent, num_epochs, steps_per_epoch=5000):
    """
    Run num epochs of training episodes.
    """

    start_time = time()
    all_results = []
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        local_step = 0
        total_reward = 0
        print(f"\nEpoch #{epoch + 1}")
        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state)
            last_variables = game.get_state().game_variables
            game.make_action(actions[action])
            local_step += 1
            done = game.is_episode_finished() or local_step > max_steps or last_variables[0] == 3
            if done:
                if not game.is_episode_finished():
                    current_variables = game.get_state().game_variables
                    reward = get_rewards(current_variables, last_variables)
                else:
                    reward = action_reward
                total_reward += reward
                next_state = np.zeros((1, int(resolution[0]), int(resolution[1]))).astype(np.float32)
                train_scores.append(total_reward)
                game.new_episode()
                local_step = 0
                total_reward = 0
            else:
                current_variables = game.get_state().game_variables
                reward = get_rewards(current_variables, last_variables)
                total_reward += reward
                next_state = preprocess(game.get_state().screen_buffer)
            agent.append_memory(state, action, reward, next_state, done)
            if global_step > agent.batch_size * 2:
                agent.train()
            global_step += 1
        all_results += train_scores
        train_scores = np.array(train_scores)
        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(train_scores.mean(), train_scores.std()),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )
        draw_results(all_results, os.path.join("results", experiment_name, "all_results.png"))
        np.save(os.path.join("results", experiment_name, "all_results.npy"), all_results)
        if save_model:
            torch.save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))
    game.close()
    return agent, game

def generate_videos(game, agent, save_path):
    """
    Generates videos of the agent playing the game.
    """

    game.set_window_visible(False)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()
    for i in range(episodes_test):
        game.new_episode()
        frames = []
        total_reward = 0
        step = 0
        current_variables = game.get_state().game_variables
        while not game.is_episode_finished() and step <= max_steps:
            # print(step, current_variables, total_reward)
            last_variables = current_variables
            screen_buf = game.get_state().screen_buffer
            if screen_buf is not None:
                kill, health, bullets, hit, _ = current_variables
                cv2.putText(screen_buf, f"kills: {int(kill)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(screen_buf, f"health: {int(health)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(screen_buf, f"bullets: {int(bullets)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(screen_buf, f"hit: {int(hit)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(screen_buf, f"steps: {int(step+1)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                frames.append(screen_buf.copy())
            if current_variables[0] >= 3:
                break
            state = preprocess(screen_buf)
            best_action_index = agent.get_action(state, mode='deterministic')
            game.make_action(actions[best_action_index])
            if not game.is_episode_finished():
                current_variables = game.get_state().game_variables
                reward = get_rewards(current_variables, last_variables)
            else:
                reward = action_reward
            total_reward += reward
            step += 1
        score = total_reward
        total_kills = current_variables[0]
        print(f"Episode #{i + 1} finished after {step} steps.")
        print('Total kills', total_kills)
        print(f"Total score: {score:.4f}")
        video_path = os.path.join(save_path, f"episode_{i + 1}.mp4")
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Saved video to {video_path}")

class QNet(nn.Module):
    """
    DQN architecture.
    """

    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(bs, -1)
        x = self.fc(x)

        return x
    
class DQNAgent:
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        epsilon=1,
        epsilon_decay=0.9995,
        epsilon_min=0.1,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile, weights_only=False, map_location='cpu').to(DEVICE)
            self.epsilon = self.epsilon_min
        else:
            print("Initializing new model")
            self.q_net = QNet(action_size).to(DEVICE)
        self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state, mode='epsilon_greedy'):
        if np.random.uniform() < self.epsilon and mode == 'epsilon_greedy':
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)
        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones
        row_idx = np.arange(self.batch_size)

        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.q_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)
        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()

    # Initialize agent
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    # Run the training
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            num_epochs=train_epochs,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished")

    game.close()

    # Generate videos of the agent playing the game
    generate_videos(
        game,
        agent,
        os.path.join("results", experiment_name, "videos"),
    )
    