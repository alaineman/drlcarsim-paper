from typing import Tuple, Dict, Text

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle


class ComplexcityEnv(AbstractEnv):
    destination_location = [0,0]
    previous_distance = 0
    vehicles_count = 5
    destination_reached = 0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "absolute": True,
                "features_range": {"x": [-270,570], "y": [-600,270], "vx": [-5, 30], "vy": [-5, 30]},
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 8, 16]
            },
            "vehicles_count": 15,
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0,
            "lane_change_reward": -0.05,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "duration": 300,
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["high_speed_reward"]], [0, 1])
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "high_speed_reward":
                 MDPVehicle.get_speed_index(self.vehicle) / (MDPVehicle.DEFAULT_TARGET_SPEEDS.size - 1),
            #"lane_change_reward": action in [0, 2],
            "on_road_reward": self.vehicle.on_road
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        #return self.vehicle.crashed or self.destination_reached == 1 or self.vehicle.lane.distance(self.vehicle.position) > 50 or \
        return self.vehicle.crashed or self.destination_reached == 1 or \
        (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius+4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 90  # [m]
        a = 5  # [m]
        delta_st = 0.2*dev  # [m]

        delta_en = dev-delta_st
        w = 2*np.pi/dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev/2], line_types=(s, c)))
        net.add_lane("ses", "se", SineLane([2+a, dev/2], [2+a, dev/2-delta_st], a, w, -np.pi/2, line_types=(c, c)))
        net.add_lane("sx", "sxs", SineLane([-2-a, -dev/2+delta_en], [-2-a, dev/2], a, w, -np.pi/2+w*delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))) #access-85
        net.add_lane("ees", "ee", SineLane([dev / 2, -2-a], [dev / 2 - delta_st, -2-a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2+a], [dev / 2, 2+a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne", SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes", "we", SineLane([-dev / 2, 2+a], [-dev / 2 + delta_st, 2+a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2-a], [-dev / 2, -2-a], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

    #Trying to add another roundabout
        p = 340
        center_1 = (0, -p)
        for lane in [0, 1]:
            net.add_lane("SE", "EX", CircularLane(center_1, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("EX", "EE", CircularLane(center_1, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("EE", "NX", CircularLane(center_1, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("NX", "NE", CircularLane(center_1, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("NE", "WX", CircularLane(center_1, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("WX", "WE", CircularLane(center_1, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("WE", "SX", CircularLane(center_1, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha), clockwise=False, line_types=line[lane]))
            net.add_lane("SX", "SE", CircularLane(center_1, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha), clockwise=False, line_types=line[lane]))

        net.add_lane("nxr", "SER", StraightLane([2, access - p], [2, access - p -1], line_types=(n, n)))
        net.add_lane("SER", "SES", StraightLane([2, access-p], [2, (dev / 2) - p], line_types=(s, c)))
        net.add_lane("SES", "SE", SineLane([2 + a, (dev / 2) - p], [2 + a, (dev / 2) - delta_st - p], a, w, (-np.pi / 2), line_types=(c, c)))

        net.add_lane("SX", "SXS", SineLane([-2 - a, -dev/2+delta_en - p], [-2 - a, (dev / 2) - p], a, w, (-np.pi / 2 + w * delta_en), line_types=(c, c)))
        net.add_lane("SXS", "SXR", StraightLane([-2, (dev / 2) - p], [-2, access - p], line_types=(n, c)))
        net.add_lane("SXR", "ner", StraightLane([-2, access - p], [-2, access -p - 1], line_types=(n, n)))

        net.add_lane("EER", "EES", StraightLane([access, -2-p], [dev / 2, -2-p], line_types=(s, c)))
        net.add_lane("EES", "EE", SineLane([dev / 2, (-2-a)-p], [dev / 2 - delta_st, (-2-a)-p], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("EX", "EXS", SineLane([-dev / 2 + delta_en, (2+a)-p], [dev / 2, (2+a)-p], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("EXS", "EXR", StraightLane([dev / 2, 2-p], [access, 2-p], line_types=(n, c)))

        net.add_lane("NER", "NES", StraightLane([-2, -access-p], [-2, (-dev / 2)-p], line_types=(s, c)))
        net.add_lane("NES", "NE", SineLane([-2 - a, (-dev / 2)-p], [-2 - a, (-dev / 2 + delta_st)-p], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("NX", "NXS", SineLane([2 + a, (dev / 2 - delta_en)-p], [2 + a, (-dev / 2)-p], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("NXS", "NXR", StraightLane([2, (-dev / 2)-p], [2, -access-p], line_types=(n, c)))

        net.add_lane("WER", "WES", StraightLane([-access, 2-p], [-dev / 2, 2-p], line_types=(s, c)))
        net.add_lane("WES", "WE", SineLane([-dev / 2, (2+a)-p], [-dev / 2 + delta_st, (2+a)-p], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("WX", "WXS", SineLane([dev / 2 - delta_en, (-2-a)-p], [-dev / 2, (-2-a)-p], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("WXS", "WXR", StraightLane([-dev / 2, -2-p], [-access, -2-p], line_types=(n, c)))

        # Making the highway

        # Circular lane to merging environment:
        center_2 = (22, -access-p-delta_st+18.5)
        net.add_lane("NXR", "merge_01", CircularLane(center_2, radii[0], np.deg2rad(180), np.deg2rad(270),
                                             clockwise=True, line_types=[c, c]))

        #parameters
        ends = [80, 50, 40, 40]  # Before, converging, merge, after
        c_1, s_1, n_1 = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        temporary_fix_x_position = 13
        temporary_fix_y_position = 13.5
        starting_position_x = (dev / 2) - 10 - temporary_fix_x_position
        starting_position_y = (2+a)+(-access-p-delta_st-22+temporary_fix_y_position) #-530
        y = [starting_position_y-8, starting_position_y-8+StraightLane.DEFAULT_WIDTH, starting_position_y-12, starting_position_y-16]
        line_type = [[s, s], [n, n], [s, n], [c, n]]
        line_type_merge = [[c_1, s_1], [n_1, s_1]]



        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([starting_position_x, starting_position_y], [starting_position_x+ends[0], starting_position_y], line_types=[s, c], forbidden=False)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2])-13, -amplitude+1.5), amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[n_1, c_1], forbidden=False)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0], line_types=[n, n], forbidden=False)
        net.add_lane("merge_01", "merge_02", ljk)
        net.add_lane("merge_02", "merge_03", lkb)
        net.add_lane("merge_03", "merge_04", lbc)


        center_bocht1 = (487, -529.5+16)
        radius_turn = [radius+4, radius, radius+8, radius+12]
        # Adding the highway lanes
        for i in range(4):
            net.add_lane("turn4", "l", StraightLane([starting_position_x - 208, y[i]], [starting_position_x-108, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("l", "m", StraightLane([starting_position_x-108, y[i]], [starting_position_x-80, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("m", "n", StraightLane([starting_position_x - 80, y[i]], [starting_position_x - 45, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("n", "a", StraightLane([starting_position_x - 45, y[i]], [starting_position_x, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("a", "merge_04", StraightLane([starting_position_x, y[i]], [starting_position_x-15+sum(ends[:2]), y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("merge_04", "b", StraightLane([starting_position_x-15+sum(ends[:2]), y[i]], [starting_position_x - 14 + sum(ends[:2]), y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("b", "c", StraightLane([starting_position_x-15+sum(ends[:2]), y[i]], [starting_position_x+sum(ends[:3]), y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("c", "d", StraightLane([starting_position_x+sum(ends[:3]), y[i]], [starting_position_x+sum(ends), y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("d", "e", StraightLane([246 - temporary_fix_x_position, y[i]], [283 - temporary_fix_x_position, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("e", "f", StraightLane([283 - temporary_fix_x_position, y[i]], [345 - temporary_fix_x_position, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("f", "g", StraightLane([345 - temporary_fix_x_position, y[i]], [385 - temporary_fix_x_position, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("g", "merge_13", StraightLane([385 - temporary_fix_x_position, y[i]], [462 - temporary_fix_x_position, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("merge_13", "h", StraightLane([462 - temporary_fix_x_position, y[i]], [500 - temporary_fix_x_position, y[i]], line_types=line_type[i], forbidden=False))
            net.add_lane("h", "turn1", CircularLane(center_bocht1, radius_turn[i], np.deg2rad(-90), np.deg2rad(0), clockwise=True, line_types=line_type[i], forbidden=False))

            if i == 1:
                net.add_lane("turn4", "l", StraightLane([starting_position_x - 208, y[i]], [starting_position_x - 108, y[i]], line_types=[n, c], forbidden=False))
                net.add_lane("l", "m", StraightLane([starting_position_x - 108, y[i]], [starting_position_x-80, y[i]], line_types=[n, s], forbidden=False))
                net.add_lane("m", "n", StraightLane([starting_position_x - 80, y[i]], [starting_position_x - 45, y[i]], line_types=[n, n], forbidden=False))
                net.add_lane("n", "a", StraightLane([starting_position_x - 45, y[i]], [starting_position_x, y[i]], line_types=[n, c], forbidden=False))
                net.add_lane("a", "merge_04", StraightLane([starting_position_x, y[i]], [starting_position_x-15+sum(ends[:2]), y[i]], line_types=[n, s], forbidden=False))
                net.add_lane("merge_04", "b", StraightLane([starting_position_x-15+sum(ends[:2]), y[i]], [starting_position_x-14+sum(ends[:2]),y[i]], line_types=[n, c], forbidden=False))
                net.add_lane("b", "c", StraightLane([starting_position_x-14+sum(ends[:2]), y[i]], [starting_position_x+sum(ends[:3]), y[i]], line_types=[n, c], forbidden=False))
                net.add_lane("c", "d", StraightLane([starting_position_x+sum(ends[:3]), y[i]], [starting_position_x+sum(ends), y[i]], line_types=[n, c], forbidden=False))
                net.add_lane("d", "e", StraightLane([246 - temporary_fix_x_position, y[i]], [283 - temporary_fix_x_position, y[i]], line_types=[n, s], forbidden=False))
                net.add_lane("e", "f", StraightLane([283 - temporary_fix_x_position, y[i]], [345 - temporary_fix_x_position, y[i]], line_types=[n, s], forbidden=False))
                net.add_lane("f", "g", StraightLane([345 - temporary_fix_x_position, y[i]], [385 - temporary_fix_x_position, y[i]], line_types=[n, c], forbidden=False))
                net.add_lane("g", "merge_13", StraightLane([385 - temporary_fix_x_position, y[i]], [462 - temporary_fix_x_position, y[i]], line_types=[n, s], forbidden=False))
                net.add_lane("merge_13", "h", StraightLane([462 - temporary_fix_x_position, y[i]], [500 - temporary_fix_x_position, y[i]], line_types=[n, c], forbidden=False))
                net.add_lane("h", "turn1", CircularLane(center_bocht1, radius_turn[i], np.deg2rad(-90), np.deg2rad(0), clockwise=True, line_types=[n, c], forbidden=False))


        # Highway from upper right to downwards right:
        line_type_upper_right_down_right = [[s, c], [s, s], [s, s], [c, n]]
        turn_2_y_position = 181+5
        turn_3_x_position = 186
        for i in range(4):
            net.add_lane("turn1", "i", StraightLane([(500 - temporary_fix_x_position+20) + (i*4), y[0]+25], [(500 - temporary_fix_x_position+20) +(i*4), y[0] - y[0]+turn_2_y_position], line_types=line_type_upper_right_down_right[i]))

        # Turn from downwards right:
        center_bocht2 = (487, -529.5+16 -y[0] + 156+5)
        for i in range(4):
            net.add_lane("i", "turn2", CircularLane(center_bocht2, radius_turn[i], np.deg2rad(0), np.deg2rad(90), clockwise=True, line_types=line_type[i]))

            if i == 1:
                net.add_lane("i", "turn2", CircularLane(center_bocht2, radius_turn[i], np.deg2rad(0), np.deg2rad(90), clockwise=True, line_types=[n, c]))
        # Highway from downwards right to downwards left:
        for i in range(4):
            net.add_lane("turn2", "j", StraightLane([487, 200+5 + (i*4)], [-turn_3_x_position, 200+5 + (i*4)], line_types=line_type_upper_right_down_right[i]))

        #Turn from downwards left:
        center_bocht3 = (-turn_3_x_position, 185)
        for i in range(4):
            net.add_lane("j", "turn3", CircularLane(center_bocht3, radius_turn[i], np.deg2rad(90), np.deg2rad(180), clockwise=True, line_types=line_type[i]))

            if i == 1:
                net.add_lane("j", "turn3", CircularLane(center_bocht3, radius_turn[i], np.deg2rad(90), np.deg2rad(180), clockwise=True, line_types=[n, c]))


        # Highway from downwards left to upwards left
        line_type_down_left_upper_left = [[c, s], [n, s], [n, s], [n, c]]
        for i in range(4):
            net.add_lane("turn3", "k", StraightLane([-turn_3_x_position-32 +(i*4), 185], [-turn_3_x_position-32 + (i*4), -access-p-4], line_types=line_type_down_left_upper_left[i]))

        # Turn from upwards left:
        center_bocht4 = (-turn_3_x_position, -access-p-3.5)
        for i in range(4):
            net.add_lane("k", "turn4", CircularLane(center_bocht4, radius_turn[i], np.deg2rad(-180), np.deg2rad(-90), clockwise=True, line_types=line_type[i]))

            if i == 1:
                net.add_lane("k", "turn4", CircularLane(center_bocht4, radius_turn[i], np.deg2rad(-180), np.deg2rad(-90), clockwise=True, line_types=[n, c]))




        # Exit lane to intersection 2
        ljk_2 = StraightLane([starting_position_x+130, starting_position_y], [starting_position_x+130 + ends[0], starting_position_y], line_types=[s, c], forbidden=False)
        lkb_2 = SineLane(ljk.position(ends[0]+131, -amplitude-4), ljk.position(sum(ends[:2]) - 13+131, (-amplitude+5.4)), amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[n_1, c_1], forbidden=False)
        net.add_lane("d", "exit_11", lkb_2)
        net.add_lane("exit_11", "exit_12", StraightLane([283-temporary_fix_x_position, starting_position_y], [283+60-temporary_fix_x_position, starting_position_y], line_types=[s, c],))
        center_3 = (24 + 318-temporary_fix_x_position, -access - p - delta_st + 5 + temporary_fix_y_position)
        net.add_lane("exit_12", "exit_13", CircularLane(center_3, radii[0], np.deg2rad(-90), np.deg2rad(0), clockwise=True, line_types=[c, c]))

        #Exit lane to roundabout 2
        lkb4 = SineLane(ljk.position(starting_position_x-130, -amplitude-4), ljk.position(starting_position_x-100, (-amplitude+4.5)), amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[n_1, c_1], forbidden=False)
        net.add_lane("l", "exit_01", lkb4)
        net.add_lane("exit_01", "exit_02", StraightLane([starting_position_x-78, starting_position_y], [starting_position_x-42, starting_position_y], line_types=[s, c]))

        center_4 = (-22, -access-p-delta_st+18.5)
        net.add_lane("exit_02", "NER", CircularLane(center_4, radii[0], np.deg2rad(-90), np.deg2rad(0), clockwise=True, line_types=[c, c]))

        #Merge lane from intersection 2:
        center_2 = (24 + 318-temporary_fix_x_position+44, -access-p-delta_st+18.5)
        net.add_lane("inter2_to_highway_1", "merge_11", CircularLane(center_2, radii[0], np.deg2rad(180), np.deg2rad(270), clockwise=True, line_types=[c, c]))

        ljk3 = StraightLane([starting_position_x+350, starting_position_y], [starting_position_x+350-40 + ends[0], starting_position_y], line_types=[s, c], forbidden=False)
        lkb3 = SineLane(ljk3.position(ends[0]-40, -amplitude), ljk3.position(sum(ends[:2]) - 13-40, -amplitude + 1.5), amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[n_1, c_1], forbidden=False)
        #lbc3 = StraightLane(lkb3.position(ends[1], 0), lkb3.position(ends[1], 0) + [ends[2], 0], line_types=[c, c], forbidden=False)
        net.add_lane("merge_11", "merge_12", ljk3)
        net.add_lane("merge_12", "merge_13", lkb3)
        #net.add_lane("merge_13", "merge_14", lbc3)






        #Intersection

        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]
        '''The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)'''

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        for corner in range(4):
            # corner = 1
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            # Incoming
            start = rotation @ np.array([(lane_width / 2), (access_length + outer_distance)])
            end = rotation @ np.array([(lane_width / 2), outer_distance])
            start[0] = start[0] + 351
            end[0] = end[0] + 351
            net.add_lane("o_a" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            #print(f"corner int1 is {corner} and start is {start}")
            #print(f"corner int1 is {corner} and end is {end}")

            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            r_center[0] = r_center[0]+351
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(-180), angle + np.radians(-90),
                                      line_types=[n, c], priority=priority, speed_limit=10))

            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            l_center[0] = l_center[0]+351
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))

            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            start[0] = start[0]+351
            end[0] = end[0]+351
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))

            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            start[0] = start[0]+351
            end[0] = end[0]+351
            net.add_lane("il" + str((corner - 1) % 4), "o_b" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        # these are connection lanes from roundabout 1 to intersection 1:
        net.add_lane("exr", "o_a1", StraightLane([access, 2], [access+70, 2], line_types=(s, c), speed_limit=10))
        net.add_lane("o_b1", "eer", StraightLane([access+70, -2], [access, -2], line_types=(n, c), speed_limit=10))


        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            # Incoming
            start = rotation @ np.array([(lane_width / 2), (access_length + outer_distance)])
            end = rotation @ np.array([(lane_width / 2), outer_distance])
            start[0] = start[0] + 351
            end[0] = end[0] + 351
            start[1] = start[1] - 340
            end[1] = end[1] - 340
            #print(f"corner int2 is {corner} and start is {start}")
            #print(f"corner int2 is {corner} and end is {end}")


            net.add_lane("O_a" + str(corner), "IR" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))

            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            r_center[0] = r_center[0]+351
            r_center[1] = r_center[1]-340
            net.add_lane("IR" + str(corner), "IL" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(-180), angle + np.radians(-90),
                                      line_types=[n, c], priority=priority, speed_limit=10))

            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            l_center[0] = l_center[0]+351
            l_center[1] = l_center[1]-340
            net.add_lane("IR" + str(corner), "IL" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))

            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            start[0] = start[0]+351
            end[0] = end[0]+351
            start[1] = start[1]-340
            end[1] = end[1]-340
            net.add_lane("IR" + str(corner), "IL" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))

            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            start[0] = start[0]+351
            end[0] = end[0]+351
            start[1] = start[1]-340
            end[1] = end[1]-340
            net.add_lane("IL" + str((corner - 1) % 4), "O_b" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))



        # connect intersections:
        x_position_intersection = 349
        net.add_lane("o_b2", "O_a0", StraightLane([x_position_intersection + 4, -111], [x_position_intersection + 4, -111 - 118], line_types=[s, c]))
        net.add_lane("O_b0", "o_a2", StraightLane([x_position_intersection, -111-118], [x_position_intersection, -111], line_types=[n, c]))

        # Connection lanes from roundabout 2 to intersection 2:
        net.add_lane("EXR", "O_a1", StraightLane([access, 2-p], [access + 70, 2-p], line_types=[s, c]))
        net.add_lane("O_b1", "EER", StraightLane([access+70, (2 - p)-4], [access, (2 - p)-4], line_types=[n, c]))

        y_position_end_inter_2 = -451

        # Connect intersection 2 to highway
        net.add_lane("O_b2", "inter2_to_highway_1", StraightLane([x_position_intersection+4, y_position_end_inter_2], [x_position_intersection+4, y_position_end_inter_2-59], line_types=[s,c]))
        net.add_lane("exit_13", "O_a2", StraightLane([x_position_intersection, y_position_end_inter_2-59], [x_position_intersection, y_position_end_inter_2], line_types=[n, c]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

        # UTurn Intersection 1 <-> Intersection 2:
        # Intersection 2 part:
        net.add_lane("O_b3", "road_to_inter1_a", StraightLane([462, -338], [472, -338], line_types=[s, c]))
        net.add_lane("road_from_inter1_a", "O_a3", StraightLane([472, -338-4], [462, -338-4], line_types=[n, c]))
        # turn to intersection 1:
        center_turn_inter2_to_inter1 = (471, -318)
        net.add_lane("road_to_inter1_a", "turn_to_inter1_a", CircularLane(center_turn_inter2_to_inter1, radii[0], np.deg2rad(-90), np.deg2rad(0), clockwise=True, line_types=[s, c]))
        net.add_lane("turn_from_inter1_a", "road_from_inter1_a", CircularLane(center_turn_inter2_to_inter1, radii[1], np.deg2rad(0), np.deg2rad(-90), clockwise=False, line_types=[n, c]))
        # Intersection 1 part:
        net.add_lane("o_b3", "road_to_inter2", StraightLane([462, 2], [472, 2], line_types=[s, c]))
        net.add_lane("road_from_inter2", "o_a3", StraightLane([472, -2], [462, -2], line_types=[n, c]))
        # Turn to intersection 2:
        center_turn_inter1_to_inter2 = (471, -22)
        net.add_lane("road_to_inter2", "turn_to_inter2", CircularLane(center_turn_inter1_to_inter2, radii[1], np.deg2rad(90), np.deg2rad(0), clockwise=False, line_types=[s, c]))
        net.add_lane("turn_from_inter2", "road_from_inter2", CircularLane(center_turn_inter1_to_inter2, radii[0], np.deg2rad(0), np.deg2rad(90), clockwise=True, line_types=[n, c]))
        #Connect both turns of intersections:
        net.add_lane("turn_to_inter2", "turn_from_inter1_a", StraightLane([472+23, 2-23], [472+23, 2-23-297], line_types=[s, c]))
        net.add_lane("turn_to_inter1_a", "turn_from_inter2", StraightLane([472+23-4, -338+18], [472+23-4, -338+18+297], line_types=[n, c]))

        #UTurn Intersection 1 <-> Roundabout 1:
        # Intersection 1 part:
        net.add_lane("o_b0", "road_to_roundbt1_a", StraightLane([349, 111], [349, access], line_types=[s, c]))
        net.add_lane("road_from_roundbt1_a", "o_a0", StraightLane([353, access], [353, 111], line_types=[n, c]))
        # Turn to roundabout 1:
        center_turn_inter1_to_roundbt1 = (349-20, access)
        net.add_lane("road_to_roundbt1_a", "turn_to_roundbt1_a", CircularLane(center_turn_inter1_to_roundbt1, radii[0], np.deg2rad(0), np.deg2rad(90), clockwise=True, line_types=[s, c]))
        net.add_lane("turn_from_roundbt1_a", "road_from_roundbt1_a", CircularLane(center_turn_inter1_to_roundbt1, radii[1], np.deg2rad(90), np.deg2rad(0), clockwise=False, line_types=[n, c]))
        # Roundabout 1 part:
        #  -
        # Turn to intersection 1:
        center_turn_roundbt1_to_inter1 = (22, access)
        net.add_lane("turn_from_inter1_b", "ser", CircularLane(center_turn_roundbt1_to_inter1, radii[0], np.deg2rad(90), np.deg2rad(180), clockwise=True, line_types=[s, c]))
        net.add_lane("sxr", "turn_to_inter1_b", CircularLane(center_turn_roundbt1_to_inter1, radii[1], np.deg2rad(180), np.deg2rad(90), clockwise=False, line_types=[n, c]))
        # Connect turns intersection 1 and roundabout:
        net.add_lane("turn_to_roundbt1_a", "turn_from_inter1_b", StraightLane([349-20, access+20], [349-20-305, access+20], line_types=[s, c]))
        net.add_lane("turn_to_inter1_b", "turn_from_roundbt1_a", StraightLane([-2+24, access+24], [-2+24+306, access+24], line_types=[n, c]))

        # UTurn Roundabout 1 <-> Roundabout 2:
        # Roundabout 1 part:
        # -
        # Turn to Roundabout 2:
        center_turn_roundbt1_to_roundbt2 = (-access, -2-20)
        net.add_lane("wxr", "turn_to_roundbt2", CircularLane(center_turn_roundbt1_to_roundbt2, radii[0], np.deg2rad(90), np.deg2rad(180), clockwise=True, line_types=[s, c]))
        net.add_lane("turn_from_roundbt2", "wer", CircularLane(center_turn_roundbt1_to_roundbt2, radii[1], np.deg2rad(180), np.deg2rad(90), clockwise=False, line_types=[n, c]))
        # Roundabout 2 part:
        # -
        # Turn to Roundabout 1:
        center_turn_roundbt2_to_roundbt1 = (-access, -2 - p +24)
        net.add_lane("turn_from_roundbt1_b", "WER", CircularLane(center_turn_roundbt2_to_roundbt1, radii[0], np.deg2rad(-180), np.deg2rad(-90), clockwise=True, line_types=[s, c]))
        net.add_lane("WXR", "turn_to_roundbt1_b", CircularLane(center_turn_roundbt2_to_roundbt1, radii[1], np.deg2rad(-90), np.deg2rad(-180), clockwise=False, line_types=[n, c]))
        # Connect Roundabout 1 and Roundabout 2:
        net.add_lane("turn_to_roundbt2", "turn_from_roundbt1_b", StraightLane([-access-20, -2-20], [-access-20, -2-20-295], line_types=[s, c]))
        net.add_lane("turn_to_roundbt1_b", "turn_from_roundbt2", StraightLane([-access-20-4, -2-p+25], [-access-20-4, -22], line_types=[n, c]))





        #road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road
        self.net = net


    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        position_deviation = 2
        speed_deviation = 2
        # Destination
        #destination = self.road.network.get_random_lane()
        #self.destination_location = destination.position(0,0)
        # print("destination is:", self.destination_location)
        # print(type(self.destination_location))
        # Ego-vehicle
        ego_lane = self.road.network.get_random_lane()
        #self.original_destination_vector_length = np.linalg.norm(np.subtract(self.destination_location, ego_lane.position(0,0)))
        #self.best_closest_distance = self.original_destination_vector_length
        # print("ego_lane location is:", ego_lane.position(0,0))
        # print(self.best_closest_distance)
        # print(ego_lane.position(0,0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(10, 0),
                                                     speed=15,
                                                     heading=ego_lane.heading_at(10))


        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        ego_vehicle.track_affiliated_lane = True
        #self.vehicle.destination_location = self.destination_location


        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        for i in range(self.vehicles_count):
            vehicle = other_vehicles_type.create_random(self.road, speed=25)

            vehicle.plan_route_to(self.road.network.get_random_destination())
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)


        possible_postions = [["turn4", "l"], ["l", "m"], ["m", "n"], ["b", "c"], ["h", "turn1"],["turn1", "i"], ["turn2", "j"], ["turn3", "k"],["sxs", "sxr"], ["ner", "nes"], ["ee", "nx"], ["wes", "we"], ["wer", "wes"], ["eer", "ees"], ["SXS", "SXR"], ["NER", "NES"], ["EE", "NX"], ["WES", "WE"], ["WER", "WES"], ["EER", "EES"],["o_a0", "ir0"], ["o_a1", "ir1"], ["o_a2", "ir2"], ["o_a3", "ir3"], ["O_a0", "IR0"], ["O_a2", "IR2"], ["O_a3", "IR3"], ["O_a1", "IR1"], ["o_b2", "O_a0"], ["O_b0", "o_a2"], ["EXR", "O_a1"], ["O_b1", "EER"],["road_to_inter2", "turn_to_inter2"], ["turn_from_inter1_a", "road_from_inter1_a"], ["sxr", "turn_to_inter1_b"], ["turn_from_roundbt1_a", "road_from_roundbt1_a"], ["WXR", "turn_to_roundbt1_b"]]
        
        #Highway
        # starting_positions_highway = [["turn4", "l"], ["l", "m"], ["m", "n"], ["b", "c"], ["h", "turn1"],["turn1", "i"], ["turn2", "j"], ["turn3", "k"]]
        # for i in range(len(starting_positions_highway)):
        #     for j in range(2, 4):
        #         vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                                    (starting_positions_highway[i][0], starting_positions_highway[i][1],j), #we, sx
        #                                                    longitudinal=20*i + self.np_random.normal()*position_deviation,
        #                                                    speed=15 + (j*1.5) + self.np_random.normal() * speed_deviation)
        #         vehicle.plan_route_to(self.road.network.get_random_destination())
        #         vehicle.randomize_behavior()
        #         self.road.vehicles.append(vehicle)
        #

        # starting_positions_rdbt1 = [["sxs", "sxr"], ["ner", "nes"], ["ee", "nx"], ["wes", "we"], ["wer", "wes"], ["eer", "ees"]]
        # starting_positions_rdbt2 = [["SXS", "SXR"], ["NER", "NES"], ["EE", "NX"], ["WES", "WE"], ["WER", "WES"], ["EER", "EES"]]
        # starting_positions_inter = [["o_a0", "ir0"], ["o_a1", "ir1"], ["o_a2", "ir2"], ["o_a3", "ir3"], ["O_a0", "IR0"], ["O_a2", "IR2"], ["O_a3", "IR3"], ["O_a1", "IR1"]]
        # starting_positions_con = [["o_b2", "O_a0"], ["O_b0", "o_a2"], ["EXR", "O_a1"], ["O_b1", "EER"]]
        # starting_positions_uturns = [["road_to_inter2", "turn_to_inter2"], ["turn_from_inter1_a", "road_from_inter1_a"], ["sxr", "turn_to_inter1_b"], ["turn_from_roundbt1_a", "road_from_roundbt1_a"], ["WXR", "turn_to_roundbt1_b"]]
        #


class ContinuousComplexcityEnv(ComplexcityEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 3, np.pi / 3],
                "longitudinal": True,
                "lateral": True,
                "dynamical": True
            },
        })
        return config

    def _reward(self, action: int) -> float:
        reward = 0
        
        #punishment for driving on the left lane
        # lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
        #     else self.vehicle.lane_index[2]
        # reward -= lane
        #
        #take the difference in radians of the heading of the car and the heading of the road
        anglediff = min(abs(self.vehicle.heading-self.vehicle.lane.lane_heading(self.vehicle.position)),abs(self.vehicle.lane.lane_heading(self.vehicle.position))+abs(self.vehicle.heading))

        #effective speed
        if self.vehicle.speed > 0:
            reward += np.cos(anglediff)*self.vehicle.speed
            
        #punishment for distance to the lane
        reward -= abs(self.vehicle.lane_distance*np.sqrt(abs(self.vehicle.lane_distance)))
        
        #scaling
        reward = reward/20
        
        #crash punishment
        if self.vehicle.crashed:
            return -10
        
        return max(reward,-5)

class DiscretizedComplexcityEnv(ContinuousComplexcityEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
            },
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [0, 5, 15],
                "actions_per_axis": (5, 9)
            },
            "offroad_terminal": False
        })
        return config

    def _reward(self, action: int) -> float:
        reward = 0
        reward += self.vehicle.speed
        reward -= self.vehicle.lane_distance**1.5
        return reward

register(
    id='complex_city-v0',
    entry_point='highway_env.envs:ComplexcityEnv',
)

register(
    id='complex_city-v1',
    entry_point='highway_env.envs:ContinuousComplexcityEnv',
)

register(
    id='complex_city-v2',
    entry_point='highway_env.envs:DiscretizedComplexcityEnv',
)
