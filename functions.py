import os

import stable_baselines3
import pprint
import numpy as np
import cv2
import gym
import highway_env
from stable_baselines3 import DQN, DDPG, TD3
from sb3_contrib import TRPO

from highway_env.vehicle.kinematics import Performance, Logger

# Used for saving the model in a xey format --  FE 11000 iterations : 11e3
def float_to_e(f):
    s = str(int(f))
    output = ""
    count = 0
    for i in range(len(s)):
        if int(s[i]) != 0:
            output += count * "0" + s[i]
            count = 0
        else:
            count += 1
    output += "e" + f"{count}"
    return output


def models(situation: str, env, alg):
    if alg.upper() == "TRPO":
        model = TRPO("MlpPolicy", env,
                     learning_rate=0.00001,  # 0.001
                     n_steps=1024,
                     batch_size=128,
                     gamma=0.99,
                     cg_max_steps=15,
                     cg_damping=0.1,
                     line_search_shrinking_factor=0.8,
                     line_search_max_iter=10,
                     n_critic_updates=10,
                     gae_lambda=0.95,
                     use_sde=False,
                     sde_sample_freq=-1,
                     normalize_advantage=True,
                     target_kl=0.01,
                     sub_sampling_factor=1,
                     policy_kwargs=None,
                     verbose=1,
                     tensorboard_log=f"tensorboard_log/{situation}_TRPO/",
                     seed=None,
                     device='cuda',
                     _init_setup_model=True)

    if alg.upper() == "DQN":
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log=f"tensorboard_log/{situation}_DQN/")
    if alg.upper() == "TD3":
        model = TD3("MlpPolicy", env,
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log=f"tensorboard_log/{situation}_TD3/")
    return model


def alg_sb3(alg, env):
    if alg.upper() == "TRPO":
        env.configure({
            'offroad_terminal': True,
            "screen_width": 1280,
            "screen_height": 560,
            "renderfps": 16,
            'simulation_frequency': 15,
            'policy_frequency': 15,
            'action': {'type': 'ContinuousAction'},
            'lateral': True,
            'longitudinal': True,
            "other_vehicles": 1,  # non-ego vehicles
            'vehicles_count': 1
        })

    if alg.upper() == "DQN":
        env.configure({
            'offroad_terminal': True,
            "screen_width": 1280,
            "screen_height": 560,
            "renderfps": 16,
            'simulation_frequency': 15,
            'policy_frequency': 15,
            'other_vehicles': 1
        })

    env.reset()

    return env


def learn(situation: str, alg: str, new_model, iterations, load_path, number_test_performance):
    env = gym.make(situation)

    if alg.upper() == "TRPO":
        env.configure({
            'offroad_terminal': True,
            "screen_width": 1280,
            "screen_height": 560,
            "renderfps": 16,
            'simulation_frequency': 15,
            'policy_frequency': 15,
            'action': {'type': 'ContinuousAction'},
            'lateral': True,
            'longitudinal': True,
            "other_vehicles": 1,  # non-ego vehicles
            'vehicles_count': 1
        })

    if alg.upper() == "DQN":
        env.configure({
            'offroad_terminal': True,
            "screen_width": 1280,
            "screen_height": 560,
            "renderfps": 16,
            'simulation_frequency': 15,
            'policy_frequency': 15,
            'other_vehicles': 1
        })

    env.reset()



    if new_model:
        model = models(situation, env, alg)
    else:
        if alg.upper() == "TRPO":
            model = TRPO.load(load_path)
        if alg.upper() == "DQN":
            model = DQN.load(load_path)
        model.set_env(env)

    for i, iter in enumerate(iterations):
        if i == 0:
            iter_round = iter
        else:
            iter_round = iter - iterations[i - 1]

        model.learn(int(iter_round))

        if new_model:
            save_path = "models/" + situation + "_" + alg + f"/{float_to_e(iter)}"
        else:
            save_path = "models/" + situation + "_" + alg + f"/{load_path.split('/')[-1]}+{float_to_e(iter)}"
        model.save(save_path)
        performace_test(env, model, save_path, i, number_test_performance)
        print(f"\n Finished learning for round {iter} of {iterations} \n")


def learn_end(situation: str, alg: str, iterations, number_test_models, number_test_performance):
    perf_results = [[pr] for pr in range(len(iterations))]

    for n in range(number_test_models):

        env = gym.make(situation)

        if alg.upper() == "TRPO":
            env.configure({
                'offroad_terminal': True,
                "screen_width": 1280,
                "screen_height": 560,
                "renderfps": 16,
                'simulation_frequency': 15,
                'policy_frequency': 15,
                'action': {'type': 'ContinuousAction'},
                'lateral': True,
                'longitudinal': True,
                "other_vehicles": 1,  # non-ego vehicles
                'vehicles_count': 1
            })

        if alg.upper() == "DQN":
            env.configure({
                'offroad_terminal': True,
                "screen_width": 1280,
                "screen_height": 560,
                "renderfps": 16,
                'simulation_frequency': 15,
                'policy_frequency': 15,
                'other_vehicles': 1
            })

        env.reset()

        model = models(situation, env, alg)

        for i, iter in enumerate(iterations):
            if i == 0:
                iter_round = iter
            else:
                iter_round = iter - iterations[i - 1]

            model.learn(int(iter_round))

            save_path = "models/" + situation + "_" + alg + f"/{float_to_e(iter)}" + "_" + f"{n+1}"
            model.save(save_path)

            perf_results[i].append(performace_test(env, model, save_path, i, number_test_performance))


    with open("models/" + situation + "_" + alg + "/" + "total" + ".txt", "w") as my_file:
        for aa in range(len(perf_results)):
            my_file.write(f"{np.mean(perf_results[aa][1:], axis=0)}")
            my_file.write("\n")
            my_file.write(f"{np.std(perf_results[aa][1:], axis=0)}")
            my_file.write("\n\n")

def performace_test(env, model, save_path, i, number_test_performance):
    perfm = Performance()
    lolly = Logger()

    number_of_runs = number_test_performance
    for f in range(number_of_runs):
        done = truncated = False
        obs, info = env.reset()
        reward = 0

        ego_car = env.controlled_vehicles[0]

        stepcounter = 0

        while (not done) and ego_car.speed > 2 and stepcounter < 800:  # 800
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            stepcounter += 1
            lolly.file(ego_car)

        perfm.add_measurement(lolly)
        lolly.clear_log()

    what = "w" if i == 0 else "a"
    with open(save_path + ".txt", what) as my_file:
        my_file.write(f"{perfm.string_rep()}")
        my_file.write(f"\n")
        my_file.write(f"{perfm.array_rep()}")
        my_file.write(f"\n\n")

    return perfm.array_rep()

def tester_end(situation: str, alg: str, load_path, number_test_models, number_test_performance):
    perf_results = []
    trained_model = load_path.split("/")[-1]
    for n in range(number_test_models):
        model_path = load_path + "/" + f"1e5_{n+1}.zip"

        env = gym.make(situation)

        if alg.upper() == "TRPO":
            env.configure({
                'offroad_terminal': True,
                "screen_width": 1280,
                "screen_height": 560,
                "renderfps": 16,
                'simulation_frequency': 15,
                'policy_frequency': 15,
                'action': {'type': 'ContinuousAction'},
                'lateral': True,
                'longitudinal': True,
                "other_vehicles": 1,  # non-ego vehicles
                'vehicles_count': 1
            })

        if alg.upper() == "DQN":
            env.configure({
                'offroad_terminal': True,
                "screen_width": 1280,
                "screen_height": 560,
                "renderfps": 16,
                'simulation_frequency': 15,
                'policy_frequency': 15,
                'other_vehicles': 1
            })

        env.reset()

        if alg.upper() == "TRPO":
            model = TRPO.load(model_path)
        if alg.upper() == "DQN":
            model = DQN.load(model_path)
        model.set_env(env)

        newpath = f"Cross evaluation/" + f"{trained_model}/" + f"{situation}/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        save_path = newpath + f"model{n+1}"

        perf_results.append(performace_test(env, model, save_path, 0, number_test_performance))

    with open(newpath + "total" + ".txt", "w") as my_file:
        my_file.write(f"{np.mean(perf_results, axis=0)}")
        my_file.write("\n")
        my_file.write(f"{np.std(perf_results, axis=0)}")
        my_file.write("\n\n")



def optimize_reward(situation: str, alg: str, iterations: float, txtfilename: str):
    p = 0
    x = 6
    total_perf = []
    array_rw = [-1 for q in range(8)]
    for i in range(1, x, 1):
        for j in range(1, x - i, 1):
            k = x - i - j

            reward_weights = [i, j, k]
            print(reward_weights)

            env = gym.make(situation)
            if alg.upper() == "TRPO":
                env.configure({
                    'offroad_terminal': True,
                    "screen_width": 1280,
                    "screen_height": 560,
                    "renderfps": 16,
                    'simulation_frequency': 15,
                    'policy_frequency': 15,
                    'action': {'type': 'ContinuousAction'},
                    'lateral': True,
                    'longitudinal': True,
                    "other_vehicles": 1,  # non-ego vehicles
                    'vehicles_count': 1,
                    'weights_array': reward_weights
                })

            if alg.upper() == "DQN":
                env.configure({
                    'offroad_terminal': True,
                    "screen_width": 1280,
                    "screen_height": 560,
                    "renderfps": 16,
                    'simulation_frequency': 15,
                    'policy_frequency': 15,
                    'other_vehicles': 1,
                    'weights_array': reward_weights
                })

            env.reset()

            env = alg_sb3(alg, gym.make(situation))

            model = models(situation, env, alg)

            model.learn(int(iterations))

            perfm = Performance()
            lolly = Logger()

            number_of_runs = 50
            for f in range(number_of_runs):
                done = truncated = False
                obs, info = env.reset()
                reward = 0

                ego_car = env.controlled_vehicles[0]

                stepcounter = 0

                while (not done) and ego_car.speed > 2 and stepcounter < 5:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    stepcounter += 1
                    lolly.file(ego_car)

                perfm.add_measurement(lolly)
                lolly.clear_log()

            feat = perfm.array_rep()

            what = "w" if p == 0 else "a"
            with open(txtfilename, what) as my_file:
                my_file.write(f"{reward_weights}")
                my_file.write(f"\n")
                my_file.write(f"{perfm.string_rep()}")
                my_file.write(f"---------------------------- \n\n")
            p += 1

            if array_rw[3] == -1:
                for l in range(8):
                    array_rw[l] = reward_weights
                feat_back = feat
            else:
                if feat[0] > feat_back[0]:
                    array_rw[0] = reward_weights
                    feat_back[0] = feat[0]
                if feat[1] < feat_back[1]:
                    array_rw[1] = reward_weights
                    feat_back[1] = feat[1]
                if feat[2] < feat_back[2]:
                    array_rw[2] = reward_weights
                    feat_back[2] = feat[2]
                if feat[3] > feat_back[3]:
                    array_rw[3] = reward_weights
                    feat_back[3] = feat[3]
                if feat[4] < feat_back[4]:
                    array_rw[4] = reward_weights
                    feat_back[4] = feat[4]
                if feat[5] > feat_back[5]:
                    array_rw[5] = reward_weights
                    feat_back[5] = feat[5]
                if feat[6] > feat_back[6]:
                    array_rw[6] = reward_weights
                    feat_back[6] = feat[6]
                if feat[7] < feat_back[7]:
                    array_rw[7] = reward_weights
                    feat_back[7] = feat[7]

            total_perf.append([reward_weights, feat])

    with open(txtfilename, "a") as my_file:
        my_file.write(f"--  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  \n")
        my_file.write(f"{array_rw}\n")
        my_file.write(f"--  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  \n")
        my_file.write(f" The HIGHEST average speed             {array_rw[0]}\n" \
                      f" The LOWEST average peak jerk          {array_rw[1]}\n" \
                      f" The LOWEST average total jerk         {array_rw[2]}\n" \
                      f" The HIGHEST average total distance    {array_rw[3]}\n" \
                      f" The LOWEST average total steering     {array_rw[4]}\n" \
                      f" The HIGHEST average duration time     {array_rw[5]}\n" \
                      f" The HIGHEST on_lane rate              {array_rw[6]}\n" \
                      f" The LOWEST collision rate of          {array_rw[7]}\n")




