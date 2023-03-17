from typing import Dict, Text

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
    #variables used
    on_road_count = 0
    off_road_count = 0
    went_offroad = 0

    #settings:
    safety = 0 #0 for standard reward function, 1 for flat punishment when crashing
    comfort = 0 #0 for standard reward, 1 for punishment when exceeding 2
    steer_punish = 1 #punish the car for steering too sharp
    road_count = 1 #reward the car more the longer it stays on the road and punish the car more the longer it stays off road
    punish_lane = 0 #punish the car for changing lanes
    go_forward = 1 #punish the car for going the wrong direction
    heading_punish = 1 #punish the car for having a large heading
    offlane_punish = 1 #punish the car for not driving in the middle of the lane


    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


class HighwayEnvRewDecent(HighwayEnv):
    #By tobias
    #Most stable reward function I have so far
    @classmethod
    #variables used
    #on_road_count = 0
    #off_road_count = 0

    #settings:
    #safety = 0 #0 for standard reward function, 1 for flat punishment when crashing
    #comfort = 0 #0 for standard reward, 1 for punishment when exceeding 2
    #steer_punish = 1 #punish the car for steering too sharp
    #road_count = 1 #reward the car more the longer it stays on the road and punish the car more the longer it stays off road
    #punish_lane = 0 #punish the car for changing lanes
    #go_forward = 1 #punish the car for going the wrong direction
    #heading_punish = 1 #punish the car for having a large heading
    #offlane_punish = 1 #punish the car for not driving in the middle of the lane
    def default_config(cls) -> dict:
        cfg = super().default_config()
        return cfg

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        reward *=15
        if self.road_count == 1:
            if int(self.vehicle.on_road) == 1:
                self.on_road_count += 1
                self.off_road_count = 0
                reward += 1.5*min(10,self.on_road_count)
            else:
                self.on_road_count = 0
                self.off_road_count += 1
                reward -= 1.5*min(10,self.off_road_count)
        if self.go_forward == 1:
            #if self.vehicle.speed <0:
            #    reward -= 30
            reward += 1.5*self.vehicle.speed
        if self.heading_punish == 1:
            head = abs(self.vehicle.heading)
            while head > 2*np.pi:
                head-=2*np.pi
            if head > np.pi:
                head = 2*np.pi-head
            #print(head)
            reward = reward -5*head**2
        if self.steer_punish == 1:
            reward -= 10*abs(action[1])**2
        if self.offlane_punish == 1:
            reward -= 5*abs(self.vehicle.lane.local_coordinates(self.vehicle.position)[1])
        if self.safety == 1:
            if int(self.vehicle.crashed) == 1:
                return -20
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

class HighwayEnvRewMinimalist(HighwayEnv):
    #By tobias
    #Experimental reward function with a minimalist approach for simplicity
    @classmethod
    #variables used
    #prev_lane = None
    #prev_acc = 0
    #on_road_count = 0
    #off_road_count = 0
    #forward_count = 0
    #backward_count = 0
    #went_offroad = 0
    
    #settings:
    #safety = 0 #0 for standard reward function, 1 for flat punishment when crashing
    #comfort = 0 #0 for standard reward, 1 for punishment when exceeding 2
    #steer_punish = 1 #punish the car for steering too sharp
    #road_count = 1 #reward the car more the longer it stays on the road and punish the car more the longer it stays off road
    #punish_lane = 0 #punish the car for changing lanes
    #go_forward = 1 #punish the car for going the wrong direction
    #heading_punish = 1 #punish the car for having a large heading
    #offlane_punish = 1 #punish the car for not driving in the middle of the lane
    def default_config(cls) -> dict:
        cfg = super().default_config()
        return cfg

    def _reward(self, action: Action) -> float:
        reward = 0
        #print(self.vehicle.position[0])
        reward += self.vehicle.speed*np.cos(self.vehicle.heading)
        if self.vehicle.speed <0 and reward>0:
            reward = -1*reward
        if int(self.vehicle.on_road) == 0:
            self.went_offroad = 1
        if self.went_offroad == 1:
            reward = 0
        if self._is_truncated() or self._is_terminated():
            #print('klaar')
            self.went_offroad = 0
        #print(self.vehicle.speed*np.cos(self.vehicle.heading))
        if self.vehicle.crashed:
            return -10
        return reward

class HighwayEnvRewV3(HighwayEnv):
    #By tobias
    #Experimental reward function with a minimalist approach for simplicity
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        return cfg

    def _reward(self, action: Action) -> float:
        reward = 0
        #print(self.vehicle.position[0])

        #reward based on speed in the x-direction, becomes negative if car is driving backwards but in the right direction
        #max speed is +- 40
        reward += self.vehicle.speed*np.cos(self.vehicle.heading)/4
        if self.vehicle.speed <0 and reward>0:
            reward = -1*reward
        if reward >0:
            reward = reward*np.sqrt(reward)
        reward -= abs(self.vehicle.lane.local_coordinates(self.vehicle.position)[1])

        
        if self.vehicle.crashed:
            return -1000
        return reward

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)

register(
    id='highway-v1',
    entry_point='highway_env.envs:HighwayEnvRewDecent',
)

register(
    id='highway-v2',
    entry_point='highway_env.envs:HighwayEnvRewMinimalist',
)

register(
    id='highway-v3',
    entry_point='highway_env.envs:HighwayEnvRewV3',
)
