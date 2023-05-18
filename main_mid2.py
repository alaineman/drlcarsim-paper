from functions import *
import highway_env 

situation = 'roundabout-v3'
alg = "DQN"                                                    # TRPO or DQN
load_path = "C:/Users/joris/Downloads/complex_city-v2_DQN"

tester_end(situation, alg, load_path, 10, 1)
