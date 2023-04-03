from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Tuple, List, Optional, Union
import numpy as np

from highway_env import utils
from highway_env.road.spline import LinearSpline2D
from highway_env.utils import wrap_to_pi, Vector, get_class_path, class_from_path


class AbstractLane(object):
    """A lane on the road, described by its central curve."""

    metaclass__ = ABCMeta
    DEFAULT_WIDTH: float = 4
    VEHICLE_LENGTH: float = 5
    length: float = 0
    line_types: List["LineType"]

    @abstractmethod
    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        """
        Convert local lane coordinates to a world position.

        :param longitudinal: longitudinal lane coordinate [m]
        :param lateral: lateral lane coordinate [m]
        :return: the corresponding world position [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        """
        Convert a world position to local lane coordinates.

        :param position: a world position [m]
        :return: the (longitudinal, lateral) lane coordinates [m]
        """
        raise NotImplementedError()

    @abstractmethod
    def lane_heading(self, position: np.ndarray) -> float:
        """
        Get the lane heading at a given world coordinate.

        :param position: world coordinate
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def heading_at(self, longitudinal: float) -> float:
        """
        Get the lane heading at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane heading [rad]
        """
        raise NotImplementedError()

    @abstractmethod
    def width_at(self, longitudinal: float) -> float:
        """
        Get the lane width at a given longitudinal lane coordinate.

        :param longitudinal: longitudinal lane coordinate [m]
        :return: the lane width [m]
        """
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: dict):
        """
        Create lane instance from config

        :param config: json dict with lane parameters
        """
        raise NotImplementedError()

    @abstractmethod
    def to_config(self) -> dict:
        """
        Write lane parameters to dict which can be serialized to json

        :return: dict of lane parameters
        """
        raise NotImplementedError()

    def on_lane(self, position: np.ndarray, longitudinal: float = None, lateral: float = None, margin: float = 0) \
            -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :return: is the position on the lane?
        """
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)
        is_on = np.abs(lateral) <= self.width_at(longitudinal) / 2 + margin and \
                -self.VEHICLE_LENGTH <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_on

    def is_reachable_from(self, position: np.ndarray) -> bool:
        """
        Whether the lane is reachable from a given world position

        :param position: the world position [m]
        :return: is the lane reachable?
        """
        if self.forbidden:
            return False
        longitudinal, lateral = self.local_coordinates(position)
        is_close = np.abs(lateral) <= 2 * self.width_at(longitudinal) and \
                   0 <= longitudinal < self.length + self.VEHICLE_LENGTH
        return is_close

    def after_end(self, position: np.ndarray, longitudinal: float = None, lateral: float = None) -> bool:
        if not longitudinal:
            longitudinal, _ = self.local_coordinates(position)
        return longitudinal > self.length - self.VEHICLE_LENGTH / 2

    def distance(self, position: np.ndarray):
        """Compute the L1 distance [m] from a position to the lane."""
        s, r = self.local_coordinates(position)
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0)

    def distance_with_heading(self, position: np.ndarray, heading: Optional[float], heading_weight: float = 1.0):
        """Compute a weighted distance in position and heading to the lane."""
        if heading is None:
            return self.distance(position)
        s, r = self.local_coordinates(position)
        angle = np.abs(self.local_angle(heading, s))
        return abs(r) + max(s - self.length, 0) + max(0 - s, 0) + heading_weight * angle

    def local_angle(self, heading: float, long_offset: float):
        """Compute non-normalised angle of heading to the lane."""
        return wrap_to_pi(heading - self.heading_at(long_offset))


class LineType:
    """A lane side line type."""

    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3


class StraightLane(AbstractLane):
    """A lane going in straight line."""

    def __init__(self,
                 start: Vector,
                 end: Vector,
                 width: float = AbstractLane.DEFAULT_WIDTH,
                 line_types: Tuple[LineType, LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        """
        New straight lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param width: the lane width [m]
        :param line_types: the type of lines on both sides of the lane
        :param forbidden: is changing to this lane forbidden
        :param priority: priority level of the lane, for determining who has right of way
        """
        self.start = np.array(start)
        self.end = np.array(end)
        self.road_vector = np.subtract(self.end, self.start)
        self.width = width
        self.length = np.linalg.norm(self.end - self.start)
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.direction = (self.end - self.start) / self.length
        self.direction_lateral = np.array([-self.direction[1], self.direction[0]])
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit
        self.colour = (255, 255, 255)  # WHITE

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return self.start + longitudinal * self.direction + lateral * self.direction_lateral

    def heading_at(self, longitudinal: float) -> float:
        road_vector = np.subtract(self.end, self.start)
        return utils.wrap_to_pi(np.arctan2(road_vector[1], road_vector[0]))

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def is_on_lane(self, position: np.ndarray) -> bool:
        proj_location = np.add(self.projection_vector(position), self.start)
        d_start = np.linalg.norm(np.subtract(proj_location, self.start))
        d_end = np.linalg.norm(np.subtract(proj_location, self.end))
        return d_start <= self.length and d_end <= self.length

    def projection_vector(self, position: np.ndarray) -> Vector:
        v = np.subtract(position, self.start)
        return np.multiply(self.road_vector, np.dot(v, self.road_vector) / np.dot(self.road_vector, self.road_vector))

    def rejection_vector(self, position: np.ndarray) -> Vector:
        road_vector = np.subtract(self.end, self.start)
        v = np.subtract(position, self.start)
        return np.subtract(v, np.multiply(road_vector, np.dot(v, road_vector) / np.dot(road_vector, road_vector)))

    def distance(self, position: np.ndarray) -> float:
        if not self.is_on_lane(position):
            d_start = np.linalg.norm(np.subtract(position, self.start))
            d_end = np.linalg.norm(np.subtract(position, self.end))
            return min(d_start, d_end)

        road_vector = np.subtract(self.end, self.start)
        d_vec = self.rejection_vector(position)
        left_normal_of_tangent = np.array(
            [road_vector[1] * -1, road_vector[0]])  # counter-clockwise orthogonal rotation
        sign = 1 if np.dot(d_vec, left_normal_of_tangent) >= 0 else -1
        return sign * np.linalg.norm(d_vec)

    def lane_heading(self, position: np.ndarray) -> float:
        road_vector = np.subtract(self.end, self.start)
        return utils.wrap_to_pi(np.arctan2(road_vector[1], road_vector[0]))

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        return float(longitudinal), float(lateral)

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "start": _to_serializable(self.start),
                "end": _to_serializable(self.end),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority
            }
        }


class SineLane(StraightLane):
    """A sinusoidal lane."""

    def __init__(self,
                 start: Vector,
                 end: Vector,
                 amplitude: float,
                 pulsation: float,
                 phase: float,
                 width: float = StraightLane.DEFAULT_WIDTH,
                 line_types: List[LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        """
        New sinusoidal lane.

        :param start: the lane starting position [m]
        :param end: the lane ending position [m]
        :param amplitude: the lane oscillation amplitude [m]
        :param pulsation: the lane pulsation [rad/m]
        :param phase: the lane initial phase [rad]
        """
        super().__init__(start, end, width, line_types, forbidden, speed_limit, priority)
        self.amplitude = amplitude
        self.pulsation = pulsation
        self.phase = phase
        self.base_vector = self.end - self.start

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return super().position(longitudinal,
                                lateral + self.amplitude * np.sin(self.pulsation * longitudinal + self.phase))

    def heading_at(self, longitudinal: float) -> float:
        return super().heading_at(longitudinal) + np.arctan(
            self.amplitude * self.pulsation * np.cos(self.pulsation * longitudinal + self.phase))

    def lane_heading(self, position: np.ndarray) -> float:
        road_vector = np.subtract(self.render_end, self.render_start)
        return utils.wrap_to_pi(np.arctan2(road_vector[1], road_vector[0]))

    @property
    def render_start(self):
        displacement = self.amplitude * np.sin(self.pulsation * 0 + self.phase)
        normalized_base_vector = np.divide(self.base_vector, np.linalg.norm(self.base_vector))
        orthonormal_base_vector = np.array([normalized_base_vector[1], normalized_base_vector[0] * -1])
        return self.start - orthonormal_base_vector * displacement

    @property
    def render_end(self):
        displacement = self.amplitude * np.sin(self.pulsation * np.linalg.norm(self.base_vector) + self.phase)
        normalized_base_vector = np.divide(self.base_vector, np.linalg.norm(self.base_vector))
        orthonormal_base_vector = np.array([normalized_base_vector[1], normalized_base_vector[0] * -1])
        return self.end - orthonormal_base_vector * displacement

    def is_on_lane(self, position: np.ndarray) -> bool:
        length = np.linalg.norm(self.render_end - self.render_start)
        proj_location = np.add(self.projection_vector(position), self.render_start)
        d_start = np.linalg.norm(np.subtract(proj_location, self.render_start))
        d_end = np.linalg.norm(np.subtract(proj_location, self.render_end))
        return d_start <= length and d_end <= length

    def projection_vector(self, position: np.ndarray) -> Vector:
        road_vector = np.subtract(self.render_end, self.render_start)
        v = np.subtract(position, self.render_start)
        return np.multiply(road_vector, np.dot(v, road_vector) / np.dot(road_vector, road_vector))

    def rejection_vector(self, position: np.ndarray) -> Vector:
        road_vector = np.subtract(self.render_end, self.render_start)
        v = np.subtract(position, self.render_start)
        return np.subtract(v, np.multiply(road_vector, np.dot(v, road_vector) / np.dot(road_vector, road_vector)))

    def distance(self, position: np.ndarray) -> float:
        if not self.is_on_lane(position):
            d_start = np.linalg.norm(np.subtract(position, self.render_start))
            d_end = np.linalg.norm(np.subtract(position, self.render_end))
            return min(d_start, d_end)

        road_vector = np.subtract(self.render_end, self.render_start)
        d_vec = self.rejection_vector(position)
        left_normal_of_tangent = np.array(
            [road_vector[1] * -1, road_vector[0]])  # counter-clockwise orthogonal rotation
        sign = 1 if np.dot(d_vec, left_normal_of_tangent) >= 0 else -1
        return sign * np.linalg.norm(d_vec)

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        longitudinal, lateral = super().local_coordinates(position)
        return longitudinal, lateral - self.amplitude * np.sin(self.pulsation * longitudinal + self.phase)

    @classmethod
    def from_config(cls, config: dict):
        config["start"] = np.array(config["start"])
        config["end"] = np.array(config["end"])
        return cls(**config)

    def to_config(self) -> dict:
        config = super().to_config()
        config.update({
            "class_path": get_class_path(self.__class__),
        })
        config["config"].update({
            "amplitude": self.amplitude,
            "pulsation": self.pulsation,
            "phase": self.phase
        })
        return config


class CircularLane(AbstractLane):
    """A lane going in circle arc."""

    def __init__(self,
                 center: Vector,
                 radius: float,
                 start_phase: float,
                 end_phase: float,
                 clockwise: bool = True,
                 width: float = AbstractLane.DEFAULT_WIDTH,
                 line_types: List[LineType] = None,
                 forbidden: bool = False,
                 speed_limit: float = 20,
                 priority: int = 0) -> None:
        super().__init__()
        self.center = np.array(center)
        self.radius = radius
        self.start_phase = utils.wrap_to_pi(start_phase)
        self.end_phase = utils.wrap_to_pi(end_phase)
        self.clockwise = clockwise
        self.direction = 1 if clockwise else -1
        self.width = width
        self.line_types = line_types or [LineType.STRIPED, LineType.STRIPED]
        self.forbidden = forbidden
        self.priority = priority
        self.speed_limit = speed_limit
        self.colour = (255, 255, 255)  # WHITE
        self.start = self.phase_to_location(self.start_phase)
        self.end = self.phase_to_location(self.end_phase)

    @property
    def phase(self):
        alpha = utils.wrap_to_pi(self.start_phase)
        beta = utils.wrap_to_pi(self.end_phase)
        if self.clockwise:
            if alpha > beta:
                return 2 * np.pi + beta - alpha
            else:
                return beta - alpha
        else:
            if alpha > beta:
                return alpha - beta
            else:
                return 2 * np.pi + alpha - beta

    @property
    def length(self):
        return self.phase * self.radius

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        return self.center + (self.radius - lateral * self.direction) * np.array([np.cos(phi), np.sin(phi)])

    def heading_at(self, longitudinal: float) -> float:
        phi = self.direction * longitudinal / self.radius + self.start_phase
        psi = phi + np.pi / 2 * self.direction
        return psi

    def width_at(self, longitudinal: float) -> float:
        return self.width

    def get_waypoints(self) -> List[Vector]:
        phase = self.end_phase - self.start_phase
        split_count = np.floor(phase / (np.pi / 2)) + 1  # for each quarter circle an additional waypoint is required
        waypoint_phases = self.split_circle(int(split_count))
        waypoints = [self.phase_to_location(phi) for phi in waypoint_phases]
        return waypoints

    def phase_to_location(self, phase: float) -> Vector:
        return self.center + self.radius * np.array([np.cos(phase * self.direction), np.sin(phase * self.direction)])

    def split_circle(self, pieces: int = 1) -> List[float]:
        phase = self.end_phase - self.start_phase
        return [self.start_phase + float(i) * (phase / pieces) for i in range(1, pieces + 1)]

    def distance_vector(self, position: np.ndarray) -> Vector:
        center_to_position_vector = np.subtract(position, self.center)
        center_pos_vec_length = np.linalg.norm(center_to_position_vector)
        resulting_vector = np.multiply(center_to_position_vector,
                                       (center_pos_vec_length - self.radius) / center_pos_vec_length)
        return resulting_vector

    def is_on_phase(self, position: np.ndarray) -> bool:
        alpha = utils.wrap_to_pi(self.start_phase)
        beta = utils.wrap_to_pi(self.end_phase)
        center_to_position_vector = np.subtract(position, self.center)
        angle = utils.wrap_to_pi(np.arctan2(center_to_position_vector[1], center_to_position_vector[0]))
        if self.clockwise and (
                (alpha > beta and (angle > alpha or angle < beta)) or (alpha < beta and beta > angle > alpha)):
            return True
        elif (not self.clockwise) and (
                (alpha > beta and beta < angle < alpha) or (alpha < beta and (angle < alpha or angle > beta))):
            return True
        return False

    #
    # Calculates the shortest distance to the road for a given location.
    # Return: the shortest distance
    #
    # Important: The distance of a position to the road, when located left of that road,
    # with respect to the lane heading, is positive. While the distance of a position to the road,
    # located to the right of that road, is given as a negative value of the Euclidean distance.
    def distance(self, position: np.ndarray) -> float:
        d_vec = self.distance_vector(position)
        if self.is_on_phase(position):
            tangent_vec = self.tangent_vector(position)
            left_normal_of_tangent = np.array(
                [tangent_vec[1] * -1, tangent_vec[0]])  # counter-clockwise orthogonal rotation
            sign = 1 if np.dot(d_vec, left_normal_of_tangent) >= 0 else -1
            return sign * np.linalg.norm(d_vec)
        d_start = np.linalg.norm(np.subtract(position, self.start))
        d_end = np.linalg.norm(np.subtract(position, self.end))
        return min(d_start, d_end)

    def tangent_vector(self, position: np.ndarray) -> Vector:
        dv = self.distance_vector(position)
        center_to_position_vector = np.subtract(position, self.center)
        is_outside_circle_arc = np.linalg.norm(center_to_position_vector) > self.radius
        if is_outside_circle_arc:
            dv = np.multiply(dv, -1)
        if self.clockwise:
            return np.array([dv[1], dv[0] * -1])  # counter-clockwise orthogonal rotation
        else:
            return np.array([dv[1] * -1, dv[0]])  # clockwise orthogonal rotation


    def lane_heading(self, position: np.ndarray) -> float:
        if self.is_on_phase(position):
            tangerine = self.tangent_vector(position)
            return utils.wrap_to_pi(np.arctan2(tangerine[1], tangerine[0]))

        d_start = np.linalg.norm(np.subtract(position, self.start))
        d_end = np.linalg.norm(np.subtract(position, self.end))
        if d_start < d_end:
            return utils.wrap_to_pi(self.start_phase + self.direction * 0.5 * np.pi)
            # heading at start
        else:
            return utils.wrap_to_pi(self.end_phase + self.direction * 0.5 * np.pi)
            # heading at end

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        delta = position - self.center
        phi = np.arctan2(delta[1], delta[0])
        phi = self.start_phase + utils.wrap_to_pi(phi - self.start_phase)
        r = np.linalg.norm(delta)
        longitudinal = self.direction * (phi - self.start_phase) * self.radius
        lateral = self.direction * (self.radius - r)
        return longitudinal, lateral

    @classmethod
    def from_config(cls, config: dict):
        config["center"] = np.array(config["center"])
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_path": get_class_path(self.__class__),
            "config": {
                "center": _to_serializable(self.center),
                "radius": self.radius,
                "start_phase": self.start_phase,
                "end_phase": self.end_phase,
                "clockwise": self.clockwise,
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority
            }
        }


class PolyLaneFixedWidth(AbstractLane):
    """
    A fixed-width lane defined by a set of points and approximated with a 2D Hermite polynomial.
    """

    def __init__(
            self,
            lane_points: List[Tuple[float, float]],
            width: float = AbstractLane.DEFAULT_WIDTH,
            line_types: Tuple[LineType, LineType] = None,
            forbidden: bool = False,
            speed_limit: float = 20,
            priority: int = 0,
    ) -> None:
        self.curve = LinearSpline2D(lane_points)
        self.length = self.curve.length
        self.width = width
        self.line_types = line_types
        self.forbidden = forbidden
        self.speed_limit = speed_limit
        self.priority = priority

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        x, y = self.curve(longitudinal)
        yaw = self.heading_at(longitudinal)
        return np.array([x - np.sin(yaw) * lateral, y + np.cos(yaw) * lateral])

    def local_coordinates(self, position: np.ndarray) -> Tuple[float, float]:
        lon, lat = self.curve.cartesian_to_frenet(position)
        return lon, lat

    def heading_at(self, longitudinal: float) -> float:
        dx, dy = self.curve.get_dx_dy(longitudinal)
        return np.arctan2(dy, dx)

    def width_at(self, longitudinal: float) -> float:
        return self.width

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def to_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "config": {
                "lane_points": _to_serializable(
                    [_to_serializable(p.position) for p in self.curve.poses]
                ),
                "width": self.width,
                "line_types": self.line_types,
                "forbidden": self.forbidden,
                "speed_limit": self.speed_limit,
                "priority": self.priority,
            },
        }


class PolyLane(PolyLaneFixedWidth):
    """
    A lane defined by a set of points and approximated with a 2D Hermite polynomial.
    """

    def __init__(
            self,
            lane_points: List[Tuple[float, float]],
            left_boundary_points: List[Tuple[float, float]],
            right_boundary_points: List[Tuple[float, float]],
            line_types: Tuple[LineType, LineType] = None,
            forbidden: bool = False,
            speed_limit: float = 20,
            priority: int = 0,
    ):
        super().__init__(
            lane_points=lane_points,
            line_types=line_types,
            forbidden=forbidden,
            speed_limit=speed_limit,
            priority=priority,
        )
        self.right_boundary = LinearSpline2D(right_boundary_points)
        self.left_boundary = LinearSpline2D(left_boundary_points)
        self._init_width()

    def width_at(self, longitudinal: float) -> float:
        if longitudinal < 0:
            return self.width_samples[0]
        elif longitudinal > len(self.width_samples) - 1:
            return self.width_samples[-1]
        else:
            return self.width_samples[int(longitudinal)]

    def _width_at_s(self, longitudinal: float) -> float:
        """
        Calculate width by taking the minimum distance between centerline and each boundary at a given s-value. This compensates indentations in boundary lines.
        """
        center_x, center_y = self.position(longitudinal, 0)
        right_x, right_y = self.right_boundary(
            self.right_boundary.cartesian_to_frenet([center_x, center_y])[0]
        )
        left_x, left_y = self.left_boundary(
            self.left_boundary.cartesian_to_frenet([center_x, center_y])[0]
        )

        dist_to_center_right = np.linalg.norm(
            np.array([right_x, right_y]) - np.array([center_x, center_y])
        )
        dist_to_center_left = np.linalg.norm(
            np.array([left_x, left_y]) - np.array([center_x, center_y])
        )

        return max(
            min(dist_to_center_right, dist_to_center_left) * 2,
            AbstractLane.DEFAULT_WIDTH,
        )

    def _init_width(self):
        """
        Pre-calculate sampled width values in about 1m distance to reduce computation during runtime. It is assumed that the width does not change significantly within 1-2m.
        Using numpys linspace ensures that min and max s-values are contained in the samples.
        """
        s_samples = np.linspace(
            0,
            self.curve.length,
            num=int(np.ceil(self.curve.length)) + 1,
        )
        self.width_samples = [self._width_at_s(s) for s in s_samples]

    def to_config(self) -> dict:
        config = super().to_config()

        ordered_boundary_points = _to_serializable(
            [_to_serializable(p.position) for p in reversed(self.left_boundary.poses)]
        )
        ordered_boundary_points += _to_serializable(
            [_to_serializable(p.position) for p in self.right_boundary.poses]
        )

        config["class_name"] = self.__class__.__name__
        config["config"]["ordered_boundary_points"] = ordered_boundary_points
        del config["config"]["width"]

        return config


def _to_serializable(arg: Union[np.ndarray, List]) -> List:
    if isinstance(arg, np.ndarray):
        return arg.tolist()
    return arg


def lane_from_config(cfg: dict) -> AbstractLane:
    return class_from_path(cfg["class_path"])(**cfg["config"])
