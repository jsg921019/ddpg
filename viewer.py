#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import numpy as np
import cv2
from env.env import Programmers_v2
from model.ActorCritic import ActorCritic

# set hyper parameters equal to when trained
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
obs_dim = 8
act_dim = 1
act_limit = 1 # np.radians(30)

env = Programmers_v2('env/map.png')
env.xycar.speed = 50
network = ActorCritic(obs_dim, act_dim, act_limit).to(device)
network.load_state_dict(torch.load('weight_0006.pth'))

state, done, ep_ret, ep_len = env.reset(0), False, 0, 0
while 1:
    frame = env.map.img.copy()
    env.draw(frame)
    cv2.imshow('viewer', frame)
    with torch.no_grad():
        state = torch.from_numpy(state).to(device)
        action = network.pi(state)
        loss = network.q(state, action)
    action = action.cpu().numpy()
    state, reward, done = env.step(action, frame=frame)
    print(reward)
    ep_ret += reward
    ep_len += 1

    if cv2.waitKey(100) == ord(' '):
        done = True

    if done:
        break
print(ep_ret)
cv2.destroyAllWindows()