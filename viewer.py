#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
학습된 Actor가 어떻게 작동하는지 볼 수 있다.(Policy를 구하는데에 Critic은 필요가 없다)
obs_dim, act_dim, act_limit를 훈련시와 동일하게 맞추고,
저장한 weight 파일을 로드시키고 실행시키면 님도 No.1 DDPG Trainer!
Spacebar로 종료.
"""

import torch
from env.env import Programmers_v2
from model.ActorCritic import Actor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set these parameters equal to when trained
obs_dim = 8
act_dim = 1
act_limit = 1

env = Programmers_v2()
actor = Actor(obs_dim, act_dim, act_limit).to(device)
actor.load_state_dict(torch.load('weight_sample.pth'))

state, done, ep_ret, ep_len = env.reset(5), False, 0, 0
env.render(draw_target_tile=True)

while not done:
    with torch.no_grad():
        state = torch.from_numpy(state).to(device)
        action = actor(state)
    action = action.cpu().numpy()
    state, reward, done, _ = env.step(action)
    env.render(draw_sensors=True, draw_target_tile=True)
    ep_ret += reward
    ep_len += 1

print('total steps:{}, total reward:{:.3f}'.format(ep_len, ep_ret))
env.close()