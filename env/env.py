#! /usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from copy import deepcopy

blue, green = (200, 0, 0), (0, 255, 0)
"""tuple: 색상 코드 지정
"""

def get_pixels_line(p1, p2):
    """두 좌표 사이의 모든 좌표값들을 구한다.

    Args:
        p1 (ndarray, dtype=int32, shape=(2,)): 첫번째 좌표.
        p2 (ndarray, dtype=int32, shape=(2,)): 두번째 좌표.

    Returns:
        (ndarray, dtype=int32, shape=(?,)): x 좌표의 값
        (ndarray, dtype=int32, shape=(?,)): y 좌표의 값.

    """
    step = np.max(np.abs(p1-p2)) + 1
    x = np.linspace(p1[0], p2[0], step, dtype=np.int32)
    y = np.linspace(p1[1], p2[1], step, dtype=np.int32)

    return x, y

def get_pixels_rect(points):
    """4개의 꼭지점 좌표로 이루어진 직사각형의 둘레를 이루는 모든 좌표를 반환.

    Args:
        points (ndarray, dtype=int32, shape=(4, 2)): 4개의 꼭지점 좌표.
            꼭지점은 시계방향 혹은 반시계방향 순으로 나열되어 있어야 한다.

    Returns:
        (ndarray, dtype=int32, shape=(?,)): x 좌표의 값
        (ndarray, dtype=int32, shape=(?,)): y 좌표의 값

    """
    p1s = points
    p2s = np.roll(points, 1, axis=0)
    x, y = [], []

    for p1, p2 in zip(p1s, p2s):
        _x, _y = get_pixels_line(p1, p2)
        x.append(_x)
        y.append(_y)

    return np.hstack(x), np.hstack(y)

def transform(points, angle, x, y):
    """입력 좌표값들을 회전/이동 변환시킨 좌표값들을 반환.

    Args:
        points (ndarray, shape=(?, 2)): 입력 좌표.
        angle (float): 반시계 방향으로 회전 시킬 radian 값.
        x (float): x축으로 이동시킬 값.
        y (float): y축으로 이동시킬 값.

    Returns:
        ndarray, dtype=int32, shape=(?, 2): 변환시킨 좌표값.

    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    translation_matrix = np.array([x, y])
    transformed = np.matmul(points, rotation_matrix) + translation_matrix

    return transformed.astype(np.int32)

class Map(object):
    """Agent가 상호작용할 맵.
    
    이미지 파일로 불러올 수 있으며 센서로부터 거리 측정의 대상이 되는 부분은 픽셀값이 검은색([0, 0, 0])이여야 한다.

    Args:
        img_path (str): 맵 이미지 경로.

    Attributes:
        img (ndarray, dtype=uint8, shape=(H, W, 3)): map의 이미지 array.
        W (int): 이미지의 너비.
        H (int): 이미지의 높이.
        init_positions (list, shape=(?, 3)): 초기 pose값 [x좌표, y좌표, radian값]의 리스트.

    """
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.H, self.W, _ = self.img.shape 
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
        """초기 pose값을 반환.

        Args:
            position (int): init_position의 몇번째 초기값으로 선택할 것인지.
                None이면 무작위로 반환. 기본값은 None.

        Returns:
            [float, float, float]: 초기값 리스트.
            int: 초기값의 인덱스값.

        """
        if position == None:
            position = np.random.randint(0, len(self.init_positions))
        return self.init_positions[position], position

class Xycar(object):
    """Bicycle model을 따르는 자이카 클래스.
    
    Attributes:
        x (float): 자이카의 x축 위치.
            x, y값은 자이카의 뒷바퀴축의 중심의 좌표이다.
        y (float): 자이카의 y축 위치.
            x, y값은 자이카의 뒷바퀴축의 중심의 좌표이다.
        yaw (float): 자이카의 heading의 radian 각도.
            x축으로 증가하는 방향으로 향할때 값이 0 이며, 반시계방향으로 증가한다.
        steer (float): 자이카의 가장 최근 조향한 값.
            -max_steer_angle ~ max_steer_angle 사이의 값이다.
        speed (float): 자이카가 가장 최근 운행한 속력값.

        width (float): 자이카의 너비.
        length (float): 자이카의 길이.
        wb (float): 자이카의 축간거리.
        max_steer_angle (float): 자이카의 최대 조향값(radian).

        box (ndarray, shape=(4, 2)): 자이카를 기준으로 한 좌표계에서의 width와 length로 구한 자이카의 꼭지점 좌표.
        hitbox (ndarray, shape=(4, 2)): 자이카를 기준으로 한 좌표계에서의 장애물과 충돌을 감지하는 히트박스의 꼭지점 좌표.

        sensor_n (int): 센서의 개수
        sensor_position (ndarray, shape=(sensor_n, 2)): 각 센서의 위치좌표.
            기본값으로 우측센서부터 좌측센서 순으로 나열되어 있고 자이카의 앞쪽에 모여있게 설정되어 있다.
        sensor_angle (ndarray, shape=(sensor_n, 2)): 자이카의 yaw값이 0일때 센서의 yaw값
            기본값으로 우측센서부터 좌측센서 순으로 나열되어 있고 -90도에서 90까지 sensor_n개 만큼 등분되게 설정되어 있다.

    """

    def __init__(self):

        # State
        self.x = None
        self.y = None
        self.yaw = None
        self.steer = 0.0
        self.speed = 0.0

        # Spec
        self.width = 27.0
        self.length = 55.0
        self.wb = 35.0
        self.max_steer_angle = np.radians(25)
        self.speed = 50.0

        # Box
        self.box = np.array([[-9.0, self.width/2], [self.length - 9.0, self.width/2],
                             [self.length - 9.0, -self.width/2], [-9.0, -self.width/2]])
        self.hitbox = np.array([[-9.0, self.width/2], [52.0, self.width/2],
                                [52.0, -self.width/2], [-9.0, -self.width/2]])
        
        # Sensors
        self.sensor_n = 7
        self.sensor_position = np.array([[46, 0] for _ in range(self.sensor_n)])
        self.sensor_angle = np.radians(np.linspace(-90, 90, self.sensor_n))
 
    def reset(self, x, y, yaw):
        """지정한 pose로 자이카를 재위치시킨다.

        Args:
            x (float): 재위치시킬 x좌표.
            y (float): 재위치시킬 y좌표.
            yaw (float): 재위치시킬 yaw(radian)값.

        """
        self.x, self.y, self.yaw = x, y, yaw
        self.yaw = self.yaw % (2.0 * np.pi)

    def step(self, steer, speed, dt):
        """주어진 조향값과 속력값으로 dt 시간만큼 이동시킨다.

        Args:
            steer (float): 자이카의 조향값으로 max_steer_angle에 이 값을 곱한 값을 사용한다.
                ex) steer이 0.8일떄 0.8*max_steer_angle 각도로 조향한다.

            speed (float): 자이카의 속력.

            dt (float): 몇 초 동안 이동시킬 것인지를 나타내는 time resolution 값.

        """
        steer *= self.max_steer_angle
        steer = np.clip(steer, -self.max_steer_angle, self.max_steer_angle)
        self.steer, self.speed = steer, speed

        self.x += speed * np.cos(self.yaw) * dt
        self.y -= speed * np.sin(self.yaw) * dt
        self.yaw += speed / self.wb * np.tan(steer) * dt
        self.yaw = self.yaw % (2.0 * np.pi)

    def get_pose(self):
        """자이카의 현재 pose를 반환한다.

        Returns:
            [float, float, float]: 차례대로 x좌표값, y좌표값, yaw값.

        """
        return [self.x, self.y, self.yaw]

    def get_box(self):
        """map 기준 좌표계에서의 자이카의 box 좌표값을 반환한다.

        Returns:
            ndarray, dtype=int32, shape=(4, 2): box의 좌표값.

        """
        return transform(self.box, self.yaw, self.x, self.y)

    def get_hitbox(self):
        """map 기준 좌표계에서의 자이카의 box 좌표값을 반환한다.

        Returns:
            ndarray, dtype=int32, shape=(4, 2): hitbox의 좌표값.

        """
        return transform(self.hitbox, self.yaw, self.x, self.y)

    def get_sensor_pose(self):
        """map 기준 좌표계에서의 자이카 센서들의 pose 값들을 반환한다.

        Returns:
            (ndarray, dtype=int32, shape=(sensor_n, 2)): 센서의 좌표값.
            (ndarray, shape=(sensor_n,)): 센서의 yaw값.

        """
        return transform(self.sensor_position, self.yaw, self.x, self.y), self.sensor_angle + self.yaw

class Reward(object):
    """자이카가 취한 행동에 따른 보상을 결정.

    기본값으로 맵을 구역별로 나누어서 이 구역을 처음 진입시 +보상을 얻고 그 이외의 경우에는 -보상을 얻는다.
    구간이 어떻게 나뉘어져 있는지는 ./env/map.py를 실행시키면 볼 수 있다.

    Attributes:

        full_tiles (list[[int]], shape=(16, 4)): 전체 16개의 타일의 인덱스 정보[y1, y2, x1, x2]를 담은 리스트.
            (x1, y1)------(x2, y1)
                |     타일      |
                |     구역      |
            (x1, y2)------(x2, y2)

        tiles (list[[int]], shape=(?, 4)): full_tiles의 부분집합.
            자이카의 초기값에 따라 full_tiles의 일부분만 사용한다.

        target_tile (list[int]): 자이카가 보상을 받을 다음 타일.

    """
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

    def reset(self, position):
        """타일을 초기화한다.

        Args:
            position (int): 자이카가 초기화한 위치 인덱스값.
                Map의 get_initial_pose에서 얻은 인덱스값이다.

        """
        self.tiles = deepcopy(self.full_tiles[:-position if position else None])
        self.target_tile = self.tiles.pop()

    def get_reward(self, x, y, done):
        """보상을 계산하여 반환한다.

        Args:
            x (float): 자이카의 x좌표값.
            y (float): 자이카의 y좌표값.
            done (bool): episode 종료 여부.
            그 외 필요시 자유롭게 추가.

        Returns:
            float: 보상값.
        
        """
        if done:
            return -3.0 # 충돌시

        i1, i2, j1, j2 = self.target_tile

        if i1 <= y < i2 and j1 <= x < j2:
            if self.tiles:
                self.target_tile = self.tiles.pop()
            else:
                self.target_tile = [0, 0, 0, 0]
            return 3.0 # target_tile 영역 안에 진입하면,
        else:
            return -0.01 # 그 이외의 경우

class Programmers_v2(object):
    """센서를 이용한 장애물 회피 주행 강화학습을 위한 환경.

    API:
        Create Environment: `env = Programmers_v2()`
        Reset Environmet: `obs = env.reset()`
        Update Environment: `next_obs, reward, done, info = env.step(action)`

    Observation:
        Type: ndarray, type=float32, shape=(obs_dim,)
        Num     Observation                       Min       Max
        0~6     Sensor Measurements(Scaled)       0         Inf
        7       Yaw                               0         2*pi rad (360 deg)
        ...     그 외 필요시 자유롭게 추가
        Note: Sensor 측정값은 300.0으로 나눈 값으로 관측값에 넘겨진다.

    Actions:
        Type: ndarray, type=float32. shape=(act_dim,)
        Num     Action                    Min                     Max
        0       Steer Angle               -1                      1
        ...     그 외 필요시 자유롭게 추가
        Note: Steer은 내부적으로 max_steer_angle를 곱해서 update를 하게 설정하였으므로
              -1에서 1사이의 값을 갖도록 의도한다.

    Reward:
        Reward 클래스에 의해 결정된다.

    Starting State:
        Map의 init_positions에서 초기값을 추출한다.

    Episode Termination:
        자이카의 hitbox에 장애물이 감지될 시 (Map의 검은 픽셀이 감지될시) done을 True로 반환한다.

    Attributes:
        map (Map): Map 객체.
        xycar (Xycar): Xycar 객체.
        reward (Reward): Reward 객체.
        dt (float): step시 시간 간격.

        hitbox_pixels (ndarray): 렌더링을 위한 캐싱.
        sensors_pixels (ndarray): 렌더링을 위한 캐싱.
    """

    def __init__(self):
        self.map = Map('env/map.png')
        self.xycar = Xycar()
        self.reward = Reward()
        self.dt = 0.1

        self.hitbox = None
        self.sensors = None

    def reset(self, position=None):
        """환경을 초기화하고 초기 관측값을 반환한다.

        Args:
            position (int): 초기 위치 리스트의 인덱스. None일시 무작위로 설정. 기본값으로 None.

        Returns:
            (ndarray, type=float32, shape=(obs_dim,)): 초기 관측값.

        """
        (x, y, yaw), position = self.map.get_initial_pose(position)
        self.xycar.reset(x, y, yaw)
        self.reward.reset(position)
        self.hitbox = self.xycar.get_hitbox()
        return self.get_observation()

    def step(self, action):
        """주어진 action으로 환경으로 update한고 다음 관측값, 보상, 종료 플래그, 정보를 반환한다.

        Args:
            action (ndarray, type=float32, shape=(act_dim,)): action 값.
                기본값으로 act_dim은 1이며 조향값을 나타낸다.

        Returns:
            (ndarray, type=float32, shape=(obs_dim,)): step 후의 관측값.
            float: 보상값.
            bool: 종료 여부.
            dict: gym API랑 맞추기 위해 넣은 것이므로 쓸 일은 없다.

        """
        steer, speed = action[0], 50.0 # 등속 운동으로 가정
        self.xycar.step(steer, speed, self.dt)
        x, y, _ = self.xycar.get_pose()
        done = self.check_collision()
        reward = self.reward.get_reward(x, y, done)

        if 0 <= y < 220 and 10 <= x < 167:
            self.reset() # 완주할시 초기화
        next_obs = self.get_observation()

        return next_obs, reward, done, {}

    def render(self, draw_hitbox=False, draw_sensors=False, draw_target_tile=False, t=None):
        """현재 환경을 OpenCV로 창으로 시각화한다.

        Args:
            draw_hitbox (bool): hitbox를 시각화할 것인지 여부. 기본값으로 False.
            draw_sensors (bool): 거리 측정값을 시각화할 것인지 여부. 기본값으로 False.
            draw_target_tile (bool): 타깃 타일을 시각화할 것인지 여부. 기본값으로 False.
            t (float): 프레임당 노출 시간. None인 겨우 dt로 계산. 기본값으로 None.

        Note:
            시각화 도중 spacebar를 누르면 종료된다.

        """

        frame = self.map.img.copy()

        # draw box
        box = self.xycar.get_box()
        cv2.fillPoly(frame, [box], blue, 4)

        # draw target_tile
        if draw_target_tile:
            i1, i2, j1, j2 = self.reward.target_tile
            frame[i1:i2, j1:j2] = np.where(frame[i1:i2, j1:j2] == 0, 0, [255,128,255])

        # draw hitbox
        if draw_hitbox:
            x, y = get_pixels_rect(self.hitbox)
            frame[y, x] = green

        #draw sensors
        if draw_sensors:
            x, y = self.sensors
            frame[y, x] = green

        # render
        cv2.imshow('programmers', frame)
        if t is None:
            t = int(1000 *self.dt)
        if cv2.waitKey(t) == ord(' '):
            self.close()
            exit()

    def close(self):
        """
        Rendering 창을 종료한다.
        """
        cv2.destroyAllWindows()

    def measure_distance(self, max_dist=None):
        """센서가 측정한 장애물과의 거리정보를 반환한다.

        Args:
            max_dist (float): 관측할 수 있는 최대 거리. 그 이상의 거리일 경우 이 값으로 측정된다.

        Returns:
            list[float], shape=(sensor_n,): 센서가 측정한 거리 정보.
        
        Note:
            반환값은 우측 센서부터 반시계방향 순이다.

        """

        if max_dist is None:
            max_dist = self.map.W + self.map.H

        measurement = []
        pixels_x, pixels_y = [], []

        sensor_position, sensor_angle  = self.xycar.get_sensor_pose()
        for p1, theta in zip(sensor_position, sensor_angle):
            p2 = p1 + np.array([max_dist * np.cos(theta), -max_dist *np.sin(theta)], dtype=np.int32)
            x, y = get_pixels_line(p1, p2)
            for i, (_x, _y) in enumerate(zip(x, y)):
                if _x == 0 or _x == self.map.W-1 or _y == 0 or _y == self.map.H-1:
                    pixels_x.append(x[:i])
                    pixels_y.append(y[:i])
                    measurement.append(max_dist)
                    break
                if np.array_equiv(self.map.img[_y, _x], 0):
                    pixels_x.append(x[:i])
                    pixels_y.append(y[:i])
                    measurement.append(np.hypot(p1[0]-_x, p1[1]-_y))
                    break
            else:
                pixels_x.append(x)
                pixels_y.append(y)
                measurement.append(max_dist)

        self.sensors = (np.hstack(pixels_x), np.hstack(pixels_y))

        return measurement

    def get_observation(self):
        """observation을 반환한다.

        Returns:
            (ndarray, type=float32, shape=(obs_dim,)): 측정된 관측값.

        """
        # Scaling을 하는 것이 학습에 도움이 된다.
        measurement = [m / 300.0 for m in self.measure_distance(max_dist=None)]
        measurement.append(self.xycar.yaw) # yaw 값을 추가한다.
        return np.array(measurement, dtype=np.float32)

    def check_collision(self):
        """Hitbox와 장애물과의 충동 여부를 판별한다.

        Returns:
            bool: 충돌 감지시 True를 반환.

        """
        self.hitbox = self.xycar.get_hitbox()
        x, y = get_pixels_rect(self.hitbox)
        try:
            if np.any(self.map.img[y, x, 1] == 0):
                return True
            else:
                return False
        except:
            return True

    # def check_goal(self):
    #     if 150 <= self.map.img[int(self.y), int(self.x), 1] < 200 :
    #         return True
    #     else:
    #         return False

    # def draw(self, frame):
    #     rect = self.xycar.get_box()
    #     cv2.fillPoly(frame, [rect], (200, 0, 0), 4)

    # def draw_hitbox(self, frame):
    #     hitbox = self.xycar.get_hitbox()
    #     x, y = get_pixels_rect(hitbox)
    #     frame[y, x] = (0, 255, 0)

if __name__ == "__main__" :
    """
    초기 위치들이 어떻게 설정되어 있는지 볼 수 있다.
    """
    env = Programmers_v2()
    for i in range(len(env.map.init_positions)):
        obs = env.reset(position=i)
        print(obs)
        env.render(draw_hitbox=True, draw_sensors=True, draw_target_tile=True, t=1000)