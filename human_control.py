import gym
import highway_env
import pprint as pp
import numpy as np
import time

from highway_env import utils
from highway_env.road.lane import CircularLane

env = gym.make('racetrack-v0')
env.configure({
    "manual_control": True,
    "action": {
        "type": "ContinuousAction"
        #"type": "DiscreteMetaAction"
    },
    "offroad_terminal": False,
    #"initial_vehicle_count": 0,
    #"spawn_probability": 0
    "vehicles_count": 5,
    "render_fps": 15,
    "simulation_frequency": 15,
    "policy_frequency": 5
})
env.reset()
pp.pprint(env.config)
score = 0
ego_car = env.controlled_vehicles[0]
while True:
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    #print(env.action_space.sample(),'\n')
    env.render()
    #print(info)
    print(ego_car.lane.distance(ego_car.position), ego_car.position, ego_car.heading)
    if isinstance(ego_car.lane, CircularLane):
        print(ego_car.lane, ego_car.lane.is_on_phase(ego_car.position), ego_car.lane.lane_heading(ego_car.position))
    # else:
    #     print('not circ lane, awh shit')
    #print(reward)
    #print(obs)
    score += reward
    #time.sleep(0.2)
print('score is',score)
