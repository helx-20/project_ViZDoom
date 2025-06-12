# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
import imageio
import matplotlib.pyplot as plt

experiment_name = "Double_dueling"
model_savefile = os.path.join("results", experiment_name, "models/model-test.pth")
mode = 'Train' # 'Train' or 'Test'
os.makedirs(os.path.join("results", experiment_name, "models"), exist_ok=True)
os.makedirs(os.path.join("results", experiment_name, "videos"), exist_ok=True)

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 4000
replay_memory_size = 10000

# NN learning settings
batch_size = 128

# Training regime
test_episodes_per_epoch = 0

# Rewards
action_reward = -0.1
killing_reward = 100
death_reward = -50
hurt_reward = -1
hit_reward = 10
shot_reward = -5

# Other parameters
# MOVE_LEFT, MOVE_RIGHT, STAY, MOVE_LEFT + ATTACK, MOVE_RIGHT + ATTACK, ATTACK
actions = [[True, False, False], [False, True, False], [False, False, False], [True, False, True], [False, True, True], [False, False, True]]
frame_repeat = 1
resolution = (30, 45) # Downsampled resolution of the input image (480*640)
episodes_to_watch = 5
max_steps = 200

if mode == 'Test':
    save_model = False
    load_model = True
    skip_learning = True
else:
    save_model = True
    load_model = False
    skip_learning = False

# Configuration file path
# config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

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
    # game.load_config(config_file_path)
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
    game.set_doom_map("map02")
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
        death = death - death_last
        hit = hit - hit_last
        hurt = health_last - health
        shot = bullets_last - bullets
    else:
        hurt = 0
        shot = 0
    reward = kills * killing_reward + hurt * hurt_reward + shot * shot_reward + hit * hit_reward + death * death_reward + action_reward

    return reward


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        total_reward = 0
        step = 0
        current_variables = game.get_state().game_variables
        while not game.is_episode_finished() and step <= max_steps:
            last_variables = current_variables
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state, mode='deterministic')
            game.make_action(actions[best_action_index], frame_repeat)
            if not game.is_episode_finished():
                current_variables = game.get_state().game_variables
                reward = get_rewards(current_variables, last_variables)
            else:
                reward = action_reward
            total_reward += reward
            step += 1
        test_scores.append(total_reward)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )

    return test_scores.mean(), test_scores.std()


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
    all_results_smooth, all_results_variance = calculate_smooth_and_variance(all_results, 10)
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
    Skip frame_repeat number of frames after each action.
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
            game.make_action(actions[action], frame_repeat)
            local_step += 1
            done = game.is_episode_finished() or local_step > max_steps
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
            if global_step > agent.batch_size:
                agent.train()
            global_step += 1

        agent.update_target_net()
        all_results += train_scores
        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        # test_mean, test_std = test(game, agent)

        draw_results(all_results, os.path.join("results", experiment_name, "all_results.png"))
        np.save(os.path.join("results", experiment_name, "all_results.npy"), all_results)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game, all_results


def generate_videos(game, agent, save_path):
    """
    Generates videos of the agent playing the game.
    """

    game.set_window_visible(False)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for i in range(episodes_to_watch):
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
                frames.append(screen_buf.copy())
            state = preprocess(screen_buf)
            best_action_index = agent.get_action(state, mode='deterministic')
            game.make_action(actions[best_action_index], frame_repeat)
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
        # Save the episode as a video
        video_path = os.path.join(save_path, f"episode_{i + 1}.mp4")
        if frames:
            imageio.mimsave(video_path, frames, fps=10)
            print(f"Saved video to {video_path}")
        else:
            print("No frames to save for this episode.")


class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.state_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(bs, -1)
        feature_size = x.shape[1]
        x1 = x[:, :int(feature_size/2)]  # input for the net to calculate the state value
        x2 = x[:, int(feature_size/2):]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (
            advantage_values - advantage_values.mean(dim=1).reshape(-1, 1)
        )

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
        epsilon_decay=0.9996,
        epsilon_min=0.05,
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
            self.target_net = torch.load(model_savefile, weights_only=False, map_location='cpu').to(DEVICE)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state, mode='epsilon_greedy'):
        if np.random.uniform() < self.epsilon and mode == 'epsilon_greedy':
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

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

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()

    # Initialize our agent with the set parameters
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game, all_results = run(
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
    