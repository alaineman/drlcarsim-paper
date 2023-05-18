from functions import *
import highway_env 

situation = 'roundabout-v3'                                    # Environment to test in
alg = "DQN"                                                    # TRPO or DQN
load_path = "C:/Users/joris/Downloads/complex_city-v2_DQN"     # Directory with models in form like "1e5_1, 1e5_2 etc

number_test_models = 10                                        # Number of models that are made
number_test_performance = 10                                   # Number of tests per model

tester_end(situation, alg, load_path, number_test_models, number_test_performance)
