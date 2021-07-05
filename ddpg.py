#! /usr/bin/env python
# -*- coding:utf-8 -*-

# import stuffs

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from env.env import Programmers_v2
from model.ActorCritic import ActorCritic
from model.ExperienceReplay import ReplayBuffer

# hyper parameters

seed=0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

obs_dim = 8
act_dim = 1
act_limit = 1 # np.radians(30)

steps_per_epoch = 2000
epochs = 100
replay_size = 100000
gamma = 0.99
polyak = 0.99
pi_lr = 0.0001
q_lr = 0.0001
batch_size = 128
start_steps = 0
update_after = 1000
update_every = 100
act_noise = 0.1
num_test_episodes = 1
max_ep_len = 2000
save_freq=1
loss_fn = nn.MSELoss()
weight_path = None # 'weight_0100.pth'

# for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# setup
env, test_env = Programmers_v2('env/map.png'), Programmers_v2('env/map.png')
replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size)
network = ActorCritic(obs_dim, act_dim, act_limit).to(device)
if weight_path is not None:
    network.load_state_dict(torch.load(weight_path))
pi_optimizer = Adam(network.pi.parameters(), lr=pi_lr)
q_optimizer = Adam(network.q.parameters(), lr=q_lr)
target_network = deepcopy(network)
network_size = 0
for param in target_network.parameters():
    network_size += np.prod(param.shape)
    param.requires_grad = False

def compute_loss_q(batch):
    state, action, reward, next_state, done = [field.to(device) for field in batch]
    q = network.q(state, action)
    with torch.no_grad():
        next_action = target_network.pi(next_state)
        next_q = target_network.q(next_state, next_action)
        target_q = reward + gamma * (1 - done) * next_q
    loss_q = loss_fn(q, target_q)

    return loss_q

def compute_loss_pi(batch):
    state = batch[0]
    state = state.to(device)
    action = network.pi(state)
    q_pi = network.q(state, action)
    return -q_pi.mean()

# Set up model saving
# logger.setup_pytorch_saver(ac)

def update(batch):

    # update q
    q_optimizer.zero_grad()
    loss_q = compute_loss_q(batch)
    loss_q.backward()
    q_optimizer.step()

    # freeze q
    for param in network.q.parameters():
        param.requires_grad = False

    # update pi
    pi_optimizer.zero_grad()
    loss_pi = compute_loss_pi(batch)
    loss_pi.backward()
    pi_optimizer.step()

    # unfreeze q
    for param in network.q.parameters():
        param.requires_grad = True

    # update target network
    with torch.no_grad():
        for param, param_target in zip(network.parameters(), target_network.parameters()):
            param_target.data.mul_(polyak)
            param_target.data.add_((1 - polyak) * param.data)

    return loss_q, loss_pi

def get_action(state, noise_scale):
    state = torch.from_numpy(state).to(device)
    action = network.get_action(state)
    action += noise_scale * np.random.randn(act_dim)
    return np.clip(action, -act_limit, act_limit)

def test_agent():

    for _ in xrange(num_test_episodes):
        state, done, ep_ret, ep_len = test_env.reset(0), False, 0, 0

        while not (done or (ep_len == max_ep_len)):
            # Take deterministic actions at test time (noise_scale=0)
            action = get_action(state, 0)
            state, reward, done = test_env.step(action)
 
            ep_ret += reward
            ep_len += 1

        print('test result : epoch=%d  steps=%d  rewards=%f'%(epoch, ep_len, ep_ret))
    print('---')


# main loop

print('\nNumber of parameters: \t %d\n'%network_size)

total_steps = steps_per_epoch * epochs
state, ep_ret, ep_len = env.reset(), 0, 0
loss_ql, loss_pil = [], []

for t in xrange(total_steps):
    
    if t > start_steps:
        action = get_action(state, act_noise)
    else:
        action = np.random.rand(act_dim)*2.0 - 2.0

    # Step the env
    next_state, reward, done = env.step(action)

    ep_ret += reward
    ep_len += 1

    replay_buffer.store(state, action, reward, next_state, done)

    state = next_state

    # End of episode handling
    if done or (ep_len == max_ep_len):
        state, ep_ret, ep_len = env.reset(), 0, 0

    # Update handling
    if replay_buffer.size >= update_after and t % update_every == 0:
        for _ in range(update_every):
            batch = replay_buffer.sample_batch(batch_size)
            loss_q, loss_pi = update(batch)
            loss_ql.append(loss_q.item())
            loss_pil.append(loss_pi.item())

    # End of epoch handling
    if (t+1) % steps_per_epoch == 0:
        epoch = (t+1) // steps_per_epoch
        print('train result : epoch=%d  loss_q=%f  loss_pi=%f'%(epoch, np.mean(loss_ql), np.mean(loss_pil)))
        loss_ql, loss_pil = [], []

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs):
            torch.save(network.state_dict(), 'weight_{:0>4}.pth'.format(epoch)) #logger.save_state({'env': env}, None)

        # Test model
        test_agent()