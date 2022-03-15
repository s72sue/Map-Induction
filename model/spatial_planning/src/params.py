from src.agents import POMCP, \
                       RandomPolicy, \
                       LandmarkPolicy, \
                       LandmarkRewardPolicy


HUMAN_TASKS = {
    "test7": ["sim_exp3/test7"],
    "test8": ["sim_exp3/test8"],
    "test9": ["sim_exp3/test9"],
    "test10": ["sim_exp3/test10"],
    "test11": ["sim_exp3/test11"],
    "test12": ["sim_exp3/test12"],

    "test7_room": ["sim_exp3_rooms/test7"],
    "test8_room": ["sim_exp3_rooms/test8"],
    "test9_room": ["sim_exp3_rooms/test9"],
    "test10_room": ["sim_exp3_rooms/test10"],
    "test11_room": ["sim_exp3_rooms/test11"],
    "test12_room": ["sim_exp3_rooms/test12"],
}

TASKS = {
    "small_simple_room": ["sim/small_simple_room"],
    "small_stim_walled": ["sim/small_stim_walled"],
    "stim1": ["sim/stim1"],
    "stim_walled": ["sim/stim_walled"],
    "four_room": ["sim/four_room"],
    "one_side": ["sim/one_side"],
    "chain": ["sim/chain"],
    "lattice": ["sim/lattice"],
    "two_room_choice_right": ["sim/two_room_choice_right"],

    "test1": ["sim_exp1/test1"],
    "test1_Reflect": ["sim_exp1/test1_Reflect"],
    "test2": ["sim_exp1/test2"],
    "test2_Reflect": ["sim_exp1/test2_Reflect"],
    "test3": ["sim_exp1/test3"],
    "test3_Reflect": ["sim_exp1/test3_Reflect"],
    "test4": ["sim_exp1/test4"],
    "test4_Reflect": ["sim_exp1/test4_Reflect"],
    "test5": ["sim_exp1/test5"],
    "test5_Reflect": ["sim_exp1/test5_Reflect"],
    "test5_Improved": ["sim_exp1/test5_Improved"],
    "test5_Reflect_Improved": ["sim_exp1/test5_Reflect_Improved"],
    "test6": ["sim_exp1/test6"],
    "test6_Reflect": ["sim_exp1/test6_Reflect"],

    "test1_room": ["sim_exp1_rooms/test1"],
    "test2_room": ["sim_exp1_rooms/test2"],
    "test3_room": ["sim_exp1_rooms/test3"],
    "test4_room": ["sim_exp1_rooms/test4"],
    "test5_room": ["sim_exp1_rooms/test5"],
    "test6_room": ["sim_exp1_rooms/test6"],

    "test7": ["stims_exp3/test7"],
    "test8": ["stims_exp3/test8"],
    "test9": ["stims_exp3/test9"],
    "test10": ["stims_exp3/test10"],
    "test11": ["stims_exp3/test11"],
    "test12": ["stims_exp3/test12"],

    "test7_room": ["stims_exp3_rooms/test7"],
    "test8_room": ["stims_exp3_rooms/test8"],
    "test9_room": ["stims_exp3_rooms/test9"],
    "test10_room": ["stims_exp3_rooms/test10"],
    "test11_room": ["stims_exp3_rooms/test11"],
    "test12_room": ["stims_exp3_rooms/test12"],

    # "test7": ["sim_exp3/test7"],
    # "test8": ["sim_exp3/test8"],
    # "test9": ["sim_exp3/test9"],
    # "test10": ["sim_exp3/test10"],
    # "test11": ["sim_exp3/test11"],
    # "test12": ["sim_exp3/test12"],

    # "test7_room": ["sim_exp3_rooms/test7"],
    # "test8_room": ["sim_exp3_rooms/test8"],
    # "test9_room": ["sim_exp3_rooms/test9"],
    # "test10_room": ["sim_exp3_rooms/test10"],
    # "test11_room": ["sim_exp3_rooms/test11"],
    # "test12_room": ["sim_exp3_rooms/test12"],

    "discovery": ["sim/two_room_choice_left",
                  "sim/two_room_choice_right",
                  "sim/two_room_choice_middle"]
}
# Optimism hyperparam based on grid 
OPT = {
    "test1": 0.00002,
    "test2": 0.00002,
    "test3": 0.00002,
    "test4": 0.00002,
    "test5": 0.00002,
    "test6": 0.00002,
    "test7": 0.0001,
    "test8": 0.0002,
    "test9": 0.00003,
    "test10": 0.0005,
    "test11": 0.000002,
    "test12": 0.00002,
}

REPLAN_STRATS = ["every_step", "mpc"]
OBSERVATION_MODES = ["fixed_radius", "fixed_radius_med", "line_of_sight", "directional_line_of_sight", "room"]


AGENTS = {"pomcp_simple": POMCP,
          "pomcp_ssp": POMCP,
          "pomcp_mle": POMCP,
          "random_policy": RandomPolicy,
          "landmark_policy": LandmarkPolicy,
          "landmark_reward_policy": LandmarkRewardPolicy
          }