import numpy as np

from highway_env.road.lane import LineType, CircularLane, StraightLane, SineLane
from highway_env.road.road import RoadNetwork, Road
from highway_env.vehicle.kinematics import Vehicle

c1 = CircularLane(np.array([2, 2]), 3, -np.pi / 2, np.pi / 2)  # clockwise
left_clock_distance = np.array([4, 4.5])
right_clock_distance = np.array([1, 1])

c2 = CircularLane(np.array([0, 0]), 3, -np.pi / 2, np.pi / 2, clockwise=False)  # counter-clockwise
right_counterclock_distance = np.array([-7, 5])
left_counterclock_distance = np.array([-1, 1])

# print("Clockwise Left:", c1.distance_vector(left_clock_distance), c1.tangent_vector(left_clock_distance),
#       c1.distance(left_clock_distance))
# print("Clockwise Right:", c1.distance_vector(right_clock_distance), c1.tangent_vector(right_clock_distance),
#       c1.distance(right_clock_distance))
# print("Left:", c2.distance_vector(left_counterclock_distance), c2.tangent_vector(left_counterclock_distance),
#       c2.distance(left_counterclock_distance))
# print("Right:", c2.distance_vector(right_counterclock_distance), c2.tangent_vector(right_counterclock_distance),
#       c2.distance(right_counterclock_distance))

s1 = StraightLane([0, 0], [3, 3])
left_position = [-1, 2]
right_position = [3, 0]

# print("Straight Left:", s1.distance(left_position))
# print("Straight right:", s1.distance(right_position))

##
center = [0, 0]  # [m]
radius = 20  # [m]
alpha = 24  # [deg]

net = RoadNetwork()
radii = [radius, radius + 4, radius + 8]
n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
line = [[c, s], [n, s], [n, c]]
for lane in range(3):
    CLE0 = CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                              clockwise=False, line_types=line[lane])
    net.add_lane("se", "ex", CLE0)
    CLE1 = CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                              clockwise=False, line_types=line[lane])
    net.add_lane("ex", "ee", CLE1)
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
dev = 85 + 15  # [m]
a = 5  # [m]
delta_st = 0.2 * dev  # [m]

delta_en = dev - delta_st
w = 2 * np.pi / dev

SL2 = StraightLane([2, access], [2, dev / 2], line_types=(s, c))
net.add_lane("ser", "ses", SL2)
net.add_lane("ses", "se", SineLane([2 + a, dev / 2], [2 + a, dev / 2 - delta_st], a, w, -np.pi / 2, line_types=(c, c)))
net.add_lane("sx", "sxs", SineLane([-2 - a, -dev / 2 + delta_en], [-2 - a, dev / 2], a, w, -np.pi / 2 + w * delta_en,
                                   line_types=(c, c)))
net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))
net.add_lane("sxr", "ser", StraightLane([-2, access], [2, access], line_types=(n, n)))

SL3 = StraightLane([access, -2], [dev / 2, -2], line_types=(s, c))
net.add_lane("eer", "ees", SL3)
SL4 = SineLane([dev / 2, -2 - a], [dev / 2 - delta_st, -2 - a], a, w, -np.pi / 2, line_types=(c, c))
net.add_lane("ees", "ee", SL4)
net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en, 2 + a], [dev / 2, 2 + a], a, w, -np.pi / 2 + w * delta_en,
                                   line_types=(c, c)))
net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))
net.add_lane("exr", "eer", StraightLane([access, 2], [access, -2], line_types=(n, n)))

net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
net.add_lane("nes", "ne",
             SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
net.add_lane("nx", "nxs", SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en,
                                   line_types=(c, c)))
net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

SL1 = StraightLane([2, -access], [-2, -access], line_types=(n, n))

net.add_lane("nxr", "ner", SL1)

net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
net.add_lane("wes", "we",
             SineLane([-dev / 2, 2 + a], [-dev / 2 + delta_st, 2 + a], a, w, -np.pi / 2, line_types=(c, c)))
net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en, -2 - a], [-dev / 2, -2 - a], a, w, -np.pi / 2 + w * delta_en,
                                   line_types=(c, c)))
net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))
net.add_lane("wxr", "wer", StraightLane([-access, -2], [-access, 2], line_types=(n, n)))

loc = [28, -6]

print(CLE1.distance(loc), CLE1.distance_vector(loc), CLE1.is_on_phase(loc), CLE1.lane_heading(loc), CLE1.tangent_vector(loc))
print(CLE0.is_on_phase(loc), CLE0.lane_heading(loc))
# print(net.get_closest_lane_index(loc))
# print(net.get_nearest_lane_index(("se", "ex", 0), loc))

