from itertools import repeat, product
from typing import Tuple, Dict, Text

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.envs.common.observation import TimeToCollisionObservation, KinematicObservation

class RacetrackEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "speed_range": [1, 5],
                "target_speeds": [0, 5, 10]
            },
            "simulation_frequency": 15,
            "policy_frequency": 15,
            "duration": 100,
            "collision_reward": -1,
            "lane_centering_cost": 4,
            "lane_centering_reward": 1,
            "action_reward": -0.3,
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
            'offroad_terminal': True
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        reward = 0
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        return {
            "lane_centering_reward": 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2),
            "action_reward": np.linalg.norm(action),
            "collision_reward": self.vehicle.crashed,
            "on_road_reward": self.vehicle.on_road,
        }

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"] or self.vehicle.lane_distance > 50

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                            speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b",
                     StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                  speed_limit=speedlimits[1]))

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -20], [120, -30],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -20], [125, -30],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2 + 5, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3 + 5, np.deg2rad(0), np.deg2rad(136), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[6]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h",
                     CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4 + 5, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4 + 5, np.deg2rad(170), np.deg2rad(58), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a",
                     CircularLane(center5, radii5 + 5, np.deg2rad(240), np.deg2rad(270), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))
            controlled_vehicle.track_affiliated_lane = True
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6 + rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6 + rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)


class RacetrackEnvModified(RacetrackEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "simulation_frequency": 15,
            "policy_frequency": 15,
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        reward = 0
        rewards = self._rewards(action)
        reward_weights = {'safety': 0.50, 'comfort': 0.25, 'efficiency': 0.25}
        for key in rewards.keys():
            reward += reward_weights[key] * rewards[key]
        # apply the penalties for not abiding by the rules
        # if self.vehicle.speed > speed_limit:
        #     reward -= (self.vehicle.speed - speed_limit) / (self.vehicle.MAX_SPEED - speed_limit)

        if not self.vehicle.on_road:
            reward -= 1
        if self.vehicle.crashed:
            reward -= 100
        #print('time for reward')
        return reward

        
    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        #(2 * self.vehicle.speed) / (
        #        self.vehicle.MAX_SPEED - self.vehicle.MIN_SPEED)  # max speed returns a reward of 1
        TTC = None
        obs_matrix = KinematicObservation(self, absolute=False, vehicles_count=self.config["other_vehicles"], normalize=False).observe()
        use_TTC = False
        glob_TTC = float('inf')
        for vehicle in range(1,len(obs_matrix)):
            x_pos = obs_matrix[vehicle][1]
            y_pos = -1 * obs_matrix[vehicle][2]
            pos_vec = [x_pos, y_pos] # this is relative when absolute = False
            vx = obs_matrix[vehicle][3]
            vy = -1 * obs_matrix[vehicle][4]
            vel_vec = [vx, vy]
            if np.dot(pos_vec, pos_vec) != 0:
                proj_pos_vel = np.multiply(np.dot(vel_vec, pos_vec) / np.dot(pos_vec, pos_vec), pos_vec)
                len_pos = np.linalg.norm(pos_vec)
                len_proj = np.linalg.norm(proj_pos_vel)

                if proj_pos_vel[0] * vel_vec[0] > 0 and proj_pos_vel[1] * vel_vec[1] > 0: # collinear so TTC infinite
                    TTC = float('Inf')
                else: 
                    TTC = len_pos / len_proj  
            else:
                TTC = float('Inf')          
            if TTC > 0: # just to be safe
                glob_TTC = min(glob_TTC, TTC)
        #print(glob_TTC)    
        if glob_TTC < 3: # only care about TTC if crash is close
            use_TTC = True


        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        if not use_TTC:   
        # all reward components are normalized (in range 0 to 1) and we take the weighted average of them in _reward
            safety_reward = 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2) 
        else:
            #(1 + self.config["lane_centering_cost"] * lateral ** 2)
            safety_reward = 0.8 * (1 - 2 / TTC) + 0.2 * 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2) 
            #(1 + 4 * self.vehicle.lane.distance(self.vehicle.position) ** 2) 
        speed_limit = 10
        if self.vehicle.speed <= speed_limit:
            efficiency_reward = self.vehicle.speed / speed_limit
        else:
            efficiency_reward = 1 - self.vehicle.speed / (self.vehicle.MAX_SPEED)


        comfort_reward = 1 - self.vehicle.jerk
        rewards_keys = ['safety', 'comfort', 'efficiency']
        rewards_values = [safety_reward, comfort_reward, efficiency_reward]
        return dict(zip(rewards_keys, rewards_values))



    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6 + rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6 + rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)

class RacetrackEnvModifiedDiscrete(RacetrackEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "simulation_frequency": 15,
            "policy_frequency": 15,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                #"features": ["presence", "x", "y", "vx", "vy", "lane_distance", "lane_heading_difference"],
            },
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [0, 15, 30],
                "actions_per_axis": (3, 5)
            }
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        reward = 0
        rewards = self._rewards(action)
        reward_weights = {'safety': 0.50, 'comfort': 0.25, 'efficiency': 0.25}
        for key in rewards.keys():
            reward += reward_weights[key] * rewards[key]
        # apply the penalties for not abiding by the rules
        # if self.vehicle.speed > speed_limit:
        #     reward -= (self.vehicle.speed - speed_limit) / (self.vehicle.MAX_SPEED - speed_limit)

        if not self.vehicle.on_road:
            reward -= 1
        if self.vehicle.crashed:
            reward -= 100
        #print('time for reward')
        return reward

        
    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        #(2 * self.vehicle.speed) / (
        #        self.vehicle.MAX_SPEED - self.vehicle.MIN_SPEED)  # max speed returns a reward of 1
        TTC = None
        obs_matrix = KinematicObservation(self, absolute=False, vehicles_count=self.config["other_vehicles"], normalize=False).observe()
        use_TTC = False
        glob_TTC = float('inf')
        for vehicle in range(1,len(obs_matrix)):
            x_pos = obs_matrix[vehicle][1]
            y_pos = -1 * obs_matrix[vehicle][2]
            pos_vec = [x_pos, y_pos] # this is relative when absolute = False
            vx = obs_matrix[vehicle][3]
            vy = -1 * obs_matrix[vehicle][4]
            vel_vec = [vx, vy]
            if np.dot(pos_vec, pos_vec) != 0:
                proj_pos_vel = np.multiply(np.dot(vel_vec, pos_vec) / np.dot(pos_vec, pos_vec), pos_vec)
                len_pos = np.linalg.norm(pos_vec)
                len_proj = np.linalg.norm(proj_pos_vel)

                if proj_pos_vel[0] * vel_vec[0] > 0 and proj_pos_vel[1] * vel_vec[1] > 0: # collinear so TTC infinite
                    TTC = float('Inf')
                else: 
                    TTC = len_pos / len_proj  
            else:
                TTC = float('Inf')          
            if TTC > 0: # just to be safe
                glob_TTC = min(glob_TTC, TTC)
        #print(glob_TTC)    
        if glob_TTC < 3: # only care about TTC if crash is close
            use_TTC = True


        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        if not use_TTC:   
        # all reward components are normalized (in range 0 to 1) and we take the weighted average of them in _reward
            safety_reward = 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2) 
        else:
            #(1 + self.config["lane_centering_cost"] * lateral ** 2)
            safety_reward = 0.8 * (1 - 2 / TTC) + 0.2 * 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2) 
            #(1 + 4 * self.vehicle.lane.distance(self.vehicle.position) ** 2) 
        speed_limit = 10
        if self.vehicle.speed <= speed_limit:
            efficiency_reward = self.vehicle.speed / speed_limit
        else:
            efficiency_reward = 1 - self.vehicle.speed / (self.vehicle.MAX_SPEED)


        comfort_reward = 1 - self.vehicle.jerk
        rewards_keys = ['safety', 'comfort', 'efficiency']
        rewards_values = [safety_reward, comfort_reward, efficiency_reward]
        return dict(zip(rewards_keys, rewards_values))



    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6 + rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6 + rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)


class RacetrackEnvPrevBest(RacetrackEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "simulation_frequency": 15,
            "policy_frequency": 15,
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        speed_limit = 20
        reward = 0
        rewards = self._rewards(action)
        reward_weights = {'safety': 0.50, 'comfort': 0.25, 'efficiency': 0.25}
        for key in rewards.keys():
            reward += reward_weights[key] * rewards[key]
        # apply the penalties for not abiding by the rules
        if self.vehicle.speed > speed_limit:
            reward -= (self.vehicle.speed - speed_limit) / (self.vehicle.MAX_SPEED - speed_limit)

        if self.vehicle.crashed or not self.vehicle.on_road:
            reward -= 1
        #print('time for reward')
        return reward

    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        # all reward components are normalized (in range 0 to 1) and we take the weighted average of them in _reward
        safety_reward = 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2)
        speed_limit = 10
        # if self.vehicle.speed <= speed_limit:
        #     efficiency_reward = self.vehicle.speed / speed_limit
        # else:
        #     efficiency_reward = 1 - self.vehicle.speed / (self.vehicle.MAX_SPEED)
        efficiency_reward = (2 * self.vehicle.speed) / (
               self.vehicle.MAX_SPEED - self.vehicle.MIN_SPEED)  # max speed returns a reward of 1


        comfort_reward = 1 - self.vehicle.jerk
        rewards_keys = ['safety', 'comfort', 'efficiency']
        rewards_values = [safety_reward, comfort_reward, efficiency_reward]
        return dict(zip(rewards_keys, rewards_values))

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = ("a", "b", rng.integers(2)) if i == 0 else \
                self.road.network.random_lane_index(rng)
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(self.road, lane_index, speed=None,
                                                                             longitudinal=rng.uniform(20, 50))

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        # Front vehicle
        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", lane_index[-1]),
                                          longitudinal=rng.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6 + rng.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles
        for i in range(rng.integers(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index(rng)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=rng.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6 + rng.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)


class DiscreteRacetrackEnv(RacetrackEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                #"features": ["presence", "x", "y", "vx", "vy", "lane_distance", "lane_heading_difference"],
            },
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [0, 15, 30],
                "actions_per_axis": (3, 5)
            }
        })
        return config

    #def _reward(self, action: int) -> float:
    #    reward = 0
    #    ego_car = self.vehicle
    #    reward -= ego_car.lane_distance**1.5
    #    if ego_car.lane_heading_difference < 1.5:
    #        reward += ego_car.speed
    #    # reward -= ego_car.lane_heading_difference
    #    return reward/25

class RacetrackEnvTest(RacetrackEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "simulation_frequency": 15,
            "policy_frequency": 15,
        })
        return config
    def _reward(self, action: np.ndarray) -> float:
        print(self.vehicle.lane.distance(self.vehicle.position),self.vehicle.lane.lane_heading(self.vehicle.position))
        return 0

register(
    id='racetrack-v0',
    entry_point='highway_env.envs:RacetrackEnv',
)

register(
    id='racetrack-v2',
    entry_point='highway_env.envs:RacetrackEnvModified'
)

register(
    id='racetrack-v5',
    entry_point='highway_env.envs:RacetrackEnvModifiedDiscrete'
)

register(
    id='racetrack-v1',
    entry_point='highway_env.envs:DiscreteRacetrackEnv',
)
register(
    id='racetrack-v3',
    entry_point='highway_env.envs:RacetrackEnvPrevBest'
)
register(
    id='racetrack-v4',
    entry_point='highway_env.envs:RacetrackEnvTest'
)

