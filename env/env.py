#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from copy import deepcopy

from torch import float32

def get_pixels_line(p1, p2):
    '''
    두 좌표 p1와 p2 사이 모든 좌표를 반환 
    '''
    step = max(abs(p1-p2)) + 1
    x = np.linspace(p1[0], p2[0], step).astype(np.int32)
    y = np.linspace(p1[1], p2[1], step).astype(np.int32)
    return x, y

def get_pixels_rect(points):
    '''
    4개의 꼭지점으로 이루어진 직사각형의 가장자리의 모든 좌표를 반화 
    '''
    p1s = points
    p2s = np.roll(points, 1, axis=0)
    x = []
    y = []
    for p1, p2 in zip(p1s, p2s):
        _x, _y = get_pixels_line(p1, p2)
        x.append(_x)
        y.append(_y)
    return np.hstack(x), np.hstack(y)

def transform(points, angle, x, y):
    '''
    좌표들을 회전/이동 변환시킨 좌표값들을 반환
    '''
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    translation_matrix = np.array([x, y])
    transformed = np.matmul(points, rotation_matrix) + translation_matrix
    return transformed.astype(np.int32)

class Map(object):

    def __init__(self, path):
        self.img = cv2.imread(path)
        self.y, self.x, _ = self.img.shape 
        self.init_positions = [[564, 50, np.radians(-90)],
                               [564, 167, np.radians(-90)],
                               [564, 271, np.radians(-90)],
                               [564, 377, np.radians(-90)],
                               [564, 505, np.radians(-135)],
                               [453, 525, np.radians(135)],
                               [403, 377, np.radians(90)],
                               [403, 271, np.radians(90)],
                               [403, 147, np.radians(135)],
                               [256, 147, np.radians(-135)],
                               [246, 272, np.radians(-90)],
                               [246, 377, np.radians(-90)],
                               [246, 505, np.radians(-135)],
                               [98 ,505, np.radians(135)],
                               [88, 377, np.radians(90)]]


 
    def get_initial_pose(self, position=None):
        if position == None:
            position = np.random.randint(0, len(self.init_positions))
        return self.init_positions[position], position

class Xycar(object):

    def __init__(self, x, y, yaw):

        # Pose
        self.x = x
        self.y = y
        self.yaw = yaw % (2.0 * np.pi)

        # Spec
        self.width = 27.0
        self.length = 55.0
        self.wb = 35.0
        self.max_steer_angle = np.radians(30)
        self.speed = 50.0

        # Box
        self.box = np.array([[-9.0, self.width/2], [self.length - 9.0, self.width/2],
                             [self.length - 9.0, -self.width/2], [-9.0, -self.width/2]])
        self.hitbox = np.array([[-9.0, self.width/2], [52.0, self.width/2],
                                [52.0, -self.width/2], [-9.0, -self.width/2]])
        
        # Sensors
        self.sensor_n = 7
        self.sensor_position = np.array([[46, 0] for _ in range(self.sensor_n)])#np.array([[20, -self.width/2 - 1], [45, -self.width/4], [45, -self.width/4], [46, 0], [45, self.width/4], [45, self.width/4], [20, self.width/2 + 1]])
        self.sensor_angle = np.radians(np.linspace(-90, 90, self.sensor_n))
 
    def step(self, steer, dt):
        steer *= self.max_steer_angle
        speed = 50.0
        steer = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)
        self.x += speed * np.cos(self.yaw) * dt
        self.y -= speed * np.sin(self.yaw) * dt
        self.yaw += speed / self.wb * np.tan(steer) * dt
        self.yaw = self.yaw % (2.0 * np.pi)

    def reset(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw

    def get_pose(self):
        return [self.x, self.y, self.yaw]

    def get_box(self):
        return transform(self.box, self.yaw, self.x, self.y)

    def get_hitbox(self):
        return transform(self.hitbox, self.yaw, self.x, self.y)

    def get_sensor_pose(self):
        return transform(self.sensor_position, self.yaw, self.x, self.y), self.sensor_angle + self.yaw

class Reward(object):

    def __init__(self):
        self.full_tiles = [[10, 220, 10, 167], [220, 325, 10, 167],
                           [325, 430, 10, 167], [430, 640, 10, 167],
                           [430, 640, 167, 325], [325, 430, 167, 325],
                           [220, 325, 167, 325], [10, 220, 167, 325],
                           [10, 220, 325, 482], [220, 325, 325, 482],
                           [325, 430, 325, 482], [430, 640, 325, 482],
                           [430, 640, 482, 640], [325, 430, 482, 640],
                           [220, 325, 482, 640], [115, 220, 482, 640]]
        self.tiles = None
        self.target_tile = None

    def reset(self, idx):
        self.tiles = deepcopy(self.full_tiles[:-idx if idx else None])
        self.target_tile = self.tiles.pop()

    def get_reward(self, x, y, done, sensors):

        if done:
            return -3.0, False

        i1, i2, j1, j2 = self.target_tile
        min_dist = np.min(sensors)
        reward = 0
        # if min_dist <= 25.0 / 300.0 and speed <= 0:
        #     reward -= 0.01
        if i1 <= y < i2 and j1 <= x < j2:
            if len(self.tiles) != 0:
                self.target_tile = self.tiles.pop()
                return reward + 1.0, False
            else:
                return reward + 1.0, True
        else:
            return reward-0.01, False

class Programmers_v2(object):

    def __init__(self, map, x=0, y=0, yaw=0):
        self.map = Map(map)
        self.xycar = Xycar(x, y, yaw)
        self.reward = Reward()
        self.dt = 0.1

    def get_state(self, frame=None):
        measurement = self.measure_distance(frame=frame, max_dist=None)
        measurement.append(self.xycar.yaw)
        return np.array(measurement, dtype=np.float32)

    def reset(self, position=None):
        (x, y, yaw), idx = self.map.get_initial_pose(position)
        self.xycar.reset(x, y, yaw)
        self.reward.reset(idx)
        return self.get_state()

    def step(self, action, frame= None):
        steer = action [0]
        self.xycar.step(steer, self.dt)
        x, y, _ = self.xycar.get_pose()
        done = self.check_collision()
        sensors = self.measure_distance()
        reward, finished = self.reward.get_reward(x, y, done, sensors)
        if finished:
            self.reset()
        state = self.get_state(frame=frame)
        return state, reward, done

    # def get_reward(self):
    #     done = self.check_collision()
    #     x, y, _ = self.xycar.get_pose()
    #     return self.reward.get_reward(x, y, done)

    def check_collision(self):
        hitbox = self.xycar.get_hitbox()
        x, y = get_pixels_rect(hitbox)
        try:
            if np.any(self.map.img[y, x, 1] == 0):
                return True
            else:
                return False
        except:
            return True

    def check_goal(self):
        if 150 <= self.map.img[int(self.y), int(self.x), 1] < 200 :
            return True
        else:
            return False

    def measure_distance(self, frame=None, max_dist = None):
        if max_dist is None:
            max_dist = self.map.x + self.map.y
        measurement = []
        sensor_position, sensor_angle  = self.xycar.get_sensor_pose()
        for p1, theta in zip(sensor_position, sensor_angle):
            p2 = p1 + max_dist * np.array([np.cos(theta), -np.sin(theta)])
            x, y = get_pixels_line(p1, p2)
            for i, (_x, _y) in enumerate(zip(x, y)):
                if _x == 0 or _x == self.map.x-1 or _y == 0 or _y == self.map.y-1:
                    if frame is not None:
                        frame[y[:i], x[:i]] = (0, 255, 0)
                    measurement.append(max_dist)
                    break
                if np.array_equiv(self.map.img[_y, _x], 0):
                    if frame is not None:
                        frame[y[:i], x[:i]] = (0, 255, 0)
                    measurement.append(np.hypot(p1[0]-_x, p1[1]-_y))
                    break
            else:
                if frame is not None:
                    frame[y, x] = (0, 255, 0)
                measurement.append(max_dist)

        return [m/300.0 for m in measurement]  #np.array(measurement, dtype=np.float32) / 300.0

    def draw(self, frame):
        rect = self.xycar.get_box()
        cv2.fillPoly(frame, [rect], (200, 0, 0), 4)
        #cv2.circle(frame, (int(self.x), int(self.y)), 5, (0,0,255), -1)
        #cv2.circle(frame, (int(self.x+ self.wb), int(self.y)), 5, (0,0,255), -1)

    def draw_hitbox(self, frame):
        hitbox = self.xycar.get_hitbox()
        x, y = get_pixels_rect(hitbox)
        frame[y, x] = (0, 255, 0)

if __name__ == "__main__" :

    env = Programmers_v2('env/map.png')
    for i in range(len(env.map.init_positions)):
        state = env.reset(position=i)
        frame = env.map.img.copy()
        #i1, i2, j1, j2 = env.reward.target_tile
        #frame[i1:i2, j1:j2] = [255,128,255]
        env.draw(frame)
        env.draw_hitbox(frame)
        measurement = env.measure_distance(frame)
        cv2.imshow('test', frame)
        if cv2.waitKey(3000) == ord(' '):
            break