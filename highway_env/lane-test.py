import numpy as np

from highway_env.road.lane import CircularLane, StraightLane
from highway_env.vehicle.kinematics import Vehicle

c1 = CircularLane(np.array([0, 0]), 3, -np.pi / 2, np.pi / 2)  # clockwise
left_clock_distance = np.array([7, 5])
right_clock_distance = np.array([1, 1])

c2 = CircularLane(np.array([0, 0]), 3, -np.pi / 2, np.pi / 2, clockwise=False)  # counter-clockwise
right_counterclock_distance = np.array([-7, 5])
left_counterclock_distance = np.array([-1, 1])

print("Clockwise Left:", c1.distance_vector(left_clock_distance), c1.tangent_vector(left_clock_distance),
      c1.distance(left_clock_distance))
print("Clockwise Right:", c1.distance_vector(right_clock_distance), c1.tangent_vector(right_clock_distance),
      c1.distance(right_clock_distance))
print("Left:", c2.distance_vector(left_counterclock_distance), c2.tangent_vector(left_counterclock_distance),
      c2.distance(left_counterclock_distance))
print("Right:", c2.distance_vector(right_counterclock_distance), c2.tangent_vector(right_counterclock_distance),
      c2.distance(right_counterclock_distance))

s1 = StraightLane([0, 0], [3, 3])
left_position = [-1, 2]
right_position = [3, 0]

print("Straight Left:", s1.distance(left_position))
print("Straight right:", s1.distance(right_position))
