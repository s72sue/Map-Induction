import julia
import numpy as np
from src.consts import *
from src.inference import *
from src.utils import vis_single_grid, write_mapcode
import copy
import time
from src.matcher import valid_multi_match, match_existing, expand, match_existing_multi
from src.proposal import gen_proposals
from collections import defaultdict
import random
from src.utils import get_cardianals, in_state_bounds, \
    find_reward, fill_obsmap, fill_reward
import math
import sys
import matplotlib.pyplot as plt

j = julia.Julia()
j.include('./src/pomdp/pomcp_planner_obs_room.jl')

class Agent:
    def __init__(self,
                 stims,
                 observation_mode="line_of_sight",
                 num_iterations=1000,
                 optimism = 0.05,
                 replan_strat="every_step",
                 room_keys=[[]],
                 *args,
                 **kwargs):

        self.stims = stims
        self.optimism = optimism
        self.replan_strat = replan_strat
        self.num_iterations = num_iterations
        self.observation_mode = observation_mode
        self.actions = ["up", "down", "left", "right"]
        self.frag_size = 5 # Larger than or equal to the expected submap resolution
        self.term = False
        self.multi = True
        self.i=0
        self.q=0
        self.room_keys = np.array(room_keys[0])

        self.room_key_masks = []
        for ki in range(np.max(self.room_keys)+1):
            mask = np.zeros(self.room_keys.shape)
            for rky in range(self.room_keys.shape[0]):
                for rkx in range(self.room_keys.shape[1]):
                    if(self.room_keys[rky, rkx] == ki):
                        mask[rky, rkx] = 1
            self.room_key_masks.append(mask)

    # Recursively descend tree to extract statistics
    def extract_tree_stats_helper(self, observation_index, partial_path, n, children, a_labels, o_labels, o_lookup_dict):
        # print("extract_tree_stats_helper", observation_index)
        paths = []
        # print(observation_index)
        child_observation = o_labels[observation_index]
        for action_index in children[observation_index]:
            new_partial_path = copy.deepcopy(partial_path)
            new_partial_path.append((a_labels[action_index], n[action_index], child_observation))
            paths.append(new_partial_path)
            child_observation_index = o_lookup_dict[action_index]
            if(len(child_observation_index)>0):
                paths += self.extract_tree_stats_helper(child_observation_index[0], new_partial_path, n, children, a_labels, o_labels, o_lookup_dict)
        return paths

    def extract_tree_stats(self, policy):
        # Get the root observation node
        total_n, children, o_labels, o_lookup, n, terminalactionnode, v, a_labels = policy

        o_lookup_dict = defaultdict(list)
        np.set_printoptions(threshold=sys.maxsize)
        for a, b, c in o_lookup:
            o_lookup_dict[a].append(c)

        # One indexing julia
        o_labels = [None, None] + o_labels
        a_labels = [None] + a_labels
        children = [None] + children

        n = [None]+list(n)
     
        return self.extract_tree_stats_helper(1, [], n, children, a_labels, o_labels, o_lookup_dict)

    def run(self):
        raise NotImplementedError

    def get_reward_squares(self, observation):
        reward_squares = []
        for y in range(observation.shape[0]):
            for x in range(observation.shape[1]):
                if(observation[y, x] == REWARD):
                    reward_squares.append((y, x))
        return reward_squares

    def remove_agent_from_observation(self, observation):
        clean_observation = copy.copy(observation)
        for y in range(observation.shape[0]):
            for x in range(observation.shape[1]):
                if(is_agent(observation[y, x])):
                    clean_observation[y, x] = EMPTY
        return clean_observation

    def populate_observation_reward(self, observation, reward_squares):
        populated_reward_obs = copy.copy(observation)
        for reward_square in reward_squares:
            populated_reward_obs[reward_square[0], reward_square[1]] = REWARD
        return populated_reward_obs

    def get_hyps(self, observation, hypLib = defaultdict(list), submapLib = defaultdict(list), reward_squares = []):

        reward_squares = list(set(reward_squares+self.get_reward_squares(observation)))
        clean_obs = self.remove_agent_from_observation(observation)
        clean_pop_obs = self.populate_observation_reward(clean_obs, reward_squares)

        if(clean_pop_obs.shape[0] >= clean_pop_obs.shape[1]):
            min_dim_y = int(clean_pop_obs.shape[0]/self.frag_size)
            min_dim_x = clean_pop_obs.shape[1]
        else:
            min_dim_y = clean_pop_obs.shape[0]
            min_dim_x = int(clean_pop_obs.shape[1]/self.frag_size)

        hypLib = defaultdict(list)
        submapLib = defaultdict(list)
        # print("Min dim: {}/{}".format(min_dim_y, min_dim_x))
        if(self.alternate_policy):
            return {}
        if(self.multi):
            prop_dict = gen_proposals(map_space = clean_pop_obs, min_dim=(min_dim_y, min_dim_x), max_dim=(min_dim_y, min_dim_x))

            # DEBUGGING
            # f=plt.figure()
            # for e, props in enumerate(prop_dict.values()):
            #     for pi, prop in enumerate(props):
            #         plt.imshow(prop)
            #         f.savefig("./temp/prop{}_{}_{}.pdf".format(self.q, e, pi), bbox_inches='tight')

            submaps = None
            submapLib = None
            hypLib, tophyps = valid_multi_match(prop_dict, clean_pop_obs, hypLib = hypLib)

            submaps_lt=None
            tophyps_lt = match_existing_multi(hypLib, clean_pop_obs)

            for tophyp_lt in tophyps_lt:
                if(all([not np.array_equal(tophyp_lt, tophyp) for tophyp in tophyps])):
                    tophyps.append(np.array(tophyp_lt))

            # Need to remove unobserved duplicates
            sorted_submap_hyps = sorted(tophyps, key=lambda submap_hyp: np.count_nonzero(submap_hyp == UNOBSERVED))


            new_tophyps = []
            reward_squares_hyps = []
            for tophyps in sorted_submap_hyps:
                novel = True
                for new_tophyp in new_tophyps:
                    if( np.count_nonzero(tophyps[new_tophyp == UNOBSERVED] != UNOBSERVED) == 0 
                        and (tophyps[tophyps != UNOBSERVED] == new_tophyp[tophyps != UNOBSERVED]).all() ):
                        novel = False

                new_reward_squares = self.get_reward_squares(tophyps)
                if(novel and new_reward_squares not in reward_squares_hyps):
                    reward_squares_hyps.append(new_reward_squares)
                    # new_submaps.append(submap)
                    new_tophyps.append(tophyps)

            belief, maps = getBelief(new_tophyps, clean_pop_obs, reward=True)
            
            # Remove zeros
            min_tophyps = []
            min_belief = []
            min_maps = []
            min_submaps = []

            print("Filtering maps...: "+str(len(new_tophyps)))
            for i in range(len(new_tophyps)):
                stitched = copy.copy(new_tophyps[i])
                stitched[observation != UNOBSERVED] = observation[observation != UNOBSERVED]
                if(belief[i] > 0 and len(self.get_reward_squares(stitched)) > 0):
                    min_tophyps.append(new_tophyps[i])
                    min_belief.append(belief[i])
                    min_maps.append(maps[i])


        else:
            prop_dict = gen_proposals(map_space = clean_pop_obs, min_dim=(min_dim_y, min_dim_x), max_dim=(min_dim_y, min_dim_x))
            


            submapLib, hypLib, submaps, tophyps = valid_match(prop_dict, clean_pop_obs, submapLib = submapLib, hypLib = hypLib)
            submaps_lt, tophyps_lt = match_existing(submapLib, hypLib, clean_pop_obs)

            print("Get hyps filter existing...")
            for tophyp_lt, submap_lt in zip(tophyps_lt, submaps_lt):
                if(all([submap_lt.shape != submap.shape or (np.array(submap_lt) != submap).any() for submap in submaps])):
                    submaps.append(np.array(submap_lt))
                    tophyps.append(np.array(tophyp_lt))


            # Need to remove unobserved duplicates
            submap_hyps = list(zip(submaps, tophyps))

            print("Sorting submap hyps...")
            sorted_submap_hyps = sorted(submap_hyps, key=lambda submap_hyp: np.count_nonzero(expand(submap_hyp[0], submap_hyp[1]) == UNOBSERVED))

            new_submaps = []
            new_tophyps = []
            reward_squares_hyps = []
            for submap, tophyps in sorted_submap_hyps:
                novel = True
                for new_submap, new_tophyp in zip(new_submaps, new_tophyps):
                    if( np.count_nonzero(expand(submap, tophyps)[expand(new_submap, new_tophyp) == UNOBSERVED] != UNOBSERVED) == 0 
                        and (expand(submap, tophyps)[expand(submap, tophyps) != UNOBSERVED] == expand(new_submap, new_tophyp)[expand(submap, tophyps) != UNOBSERVED]).all() ):
                        novel = False

                new_reward_squares = self.get_reward_squares(expand(submap, tophyps))
                if(novel and new_reward_squares not in reward_squares_hyps):
                    reward_squares_hyps.append(new_reward_squares)
                    new_submaps.append(submap)
                    new_tophyps.append(tophyps)

            belief, maps = getBelief([expand(s, h) for s, h in zip(new_submaps, new_tophyp)], clean_pop_obs, reward=True)
            
            # Remove zeros
            min_submaps = []
            min_tophyps = []
            min_belief = []
            min_maps = []

            print("Filtering maps...: "+str(len(new_submaps)))
            for i in range(len(new_submaps)):
                stitched = copy.copy(expand(new_submaps[i], new_tophyps[i]))
                stitched[observation != UNOBSERVED] = observation[observation != UNOBSERVED]
                if(belief[i] > 0 and len(self.get_reward_squares(stitched)) > 0):
                    min_submaps.append(new_submaps[i])
                    min_tophyps.append(new_tophyps[i])
                    min_belief.append(belief[i])
                    min_maps.append(maps[i])

        # Add reward decay map
        min_reward_maps = [fill_reward(smap, observation)+[None, None] for smap in min_maps]+[None, None]
        min_fill_obsmap, min_fill_key = fill_obsmap(observation)

        f=plt.figure()
        plt.imshow(observation)
        f.savefig("./temp/obs{}.pdf".format(self.q), bbox_inches='tight')

        # for e, min_map in enumerate(min_maps):
        #     plt.imshow(min_map)
        #     f.savefig("./temp/hyp{}_{}.pdf".format(self.q, e), bbox_inches='tight')

        self.q+=1
        print("Num hyps: "+str(len(min_maps)))
        return {"submapLib": submapLib,
                "hypLib": hypLib,
                "submaps": min_submaps,
                "tophyps": min_tophyps,
                "reward_squares": reward_squares,
                "reward_maps": min_reward_maps,
                "beliefs": min_belief,
                "maps": min_maps,
                "obs_reward_map": min_fill_obsmap,
                "obs_reward_key": min_fill_key}

    def get_actions(self, current_observed_state, current_observation, run_infos):
        raise NotImplementedError

    def run(self):
        # TODO: Split up and move to Agent superclass
        actions = []
        rewards = []
        states = []
        observed_states = []
        run_infos = []

        for stim in self.stims:
            sy, sx = stim.shape
            current_state = stim
            current_observed_state = j.init_observed_state(stim, self.observation_mode, self.room_keys, self.room_key_masks)
            current_observation = current_observed_state
            states.append(current_state)
            observed_states.append(current_observed_state)

            # Get map hypotheses from observations
            if(len(run_infos) > 0):# TODO: Clean up
                run_info = self.get_hyps(current_observed_state,
                                         hypLib=run_infos[-1]['hypLib'],
                                         submapLib=run_infos[-1]['submapLib'],
                                         reward_squares=[])
            else:
                run_info = self.get_hyps(current_observed_state)
            run_infos.append(run_info)

            it = 0
            previous_reward = None
            while it < self.num_iterations and not self.term:
                print("Planning iteration: "+str(it))
                run_info_extras = {}
                start_time = time.time()
                pred_actions, run_info_extras = self.get_actions(current_observed_state, current_observation, previous_reward, run_infos)

                print("Solution Time: "+str(time.time()-start_time))
                for action in pred_actions:
                    next_state, new_observed_state, reward =  \
                        j.next_state(current_state,
                                     current_observed_state,
                                     action,
                                     self.observation_mode,
                                     self.optimism,
                                     self.room_keys,
                                     self.room_key_masks)

                    current_observation = j.init_observed_state(next_state, self.observation_mode, self.room_keys, self.room_key_masks)

                    states.append(next_state)
                    observed_states.append(new_observed_state)
                    actions.append(action)
                    rewards.append(reward)
                    previous_reward = reward

                    current_state = next_state
                    past_observed_state = copy.copy(current_observed_state)
                    current_observed_state = copy.copy(new_observed_state)

                    # # Get map hypotheses from observations
                    # run_info = self.get_hyps(current_observed_state,
                    #                          hypLib=run_info['hypLib'],
                    #                          submapLib=run_info['submapLib'],
                    #                          reward_squares=run_info['reward_squares'])
                    run_info={}
                    run_info.update(run_info_extras)
                    run_infos.append(run_info)

                    it += 1

                    if(self.replan_strat == "every_step" or 
                        np.count_nonzero(past_observed_state == UNOBSERVED) !=
                        np.count_nonzero(current_observed_state == UNOBSERVED)
                            or np.count_nonzero(current_state == REWARD) == 0):
                        break

                if(np.count_nonzero(current_state == REWARD) == 0):
                    break

        return states, observed_states, actions, rewards, run_infos


class RandomPolicy(Agent):
    def __init__(self, stims, **kwargs):
        self.stims = stims
        super(RandomPolicy, self).__init__(stims, **kwargs)

    def get_actions(self, current_observed_state, current_observation, previous_reward, run_infos):
        return [random.choice(self.actions)], {}


class LandmarkPolicy(Agent):
    def __init__(self, stims, **kwargs):
        self.stims = stims
        self.memory_action_map = []
        super(LandmarkPolicy, self).__init__(stims, **kwargs)

    def get_in_memory(self, observation):
        for (rem, action) in list(self.memory_action_map):
            if(np.count_nonzero(rem-observation) == 0):
                return (rem, action)
        return None, None

    def get_actions(self, current_observed_state, current_observation, previous_reward, run_infos):
        instance, action = self.get_in_memory(current_observation)
        if(instance is not None):
            return [action], {}
        else:
            random_action = random.choice(self.actions)
            self.memory_action_map.append((current_observation, random_action))
            return [random_action], {}


class LandmarkRewardPolicy(Agent):
    def __init__(self, stims, reward_steps_k=20, **kwargs):
        self.stims = stims
        self.unused_memory_action_map = []
        self.memory_action_map = []
        self.reward_steps_k = reward_steps_k
        super(LandmarkRewardPolicy, self).__init__(stims, **kwargs)

    def get_in_memory(self, observation):
        for (rem, action) in list(self.memory_action_map):
            if(np.count_nonzero(rem-observation) == 0):
                return (rem, action)
        return None, None

    def find_nearest_unobserved(self, current_observed_state):
        # Get the agent state
        agent_pos = None
        for i in range(current_observed_state.shape[0]):
            for j in range(current_observed_state.shape[1]):
                if(is_agent(current_observed_state[i, j])):
                    agent_pos = (i, j)
        assert agent_pos is not None
        explored = []
        Q = [(agent_pos, [])]
        while(len(Q) > 0):
            q, path = Q[0]
            explored.append(q)
            del Q[0]
            if(current_observed_state[q[0], q[1]] == UNOBSERVED or current_observed_state[q[0], q[1]] == REWARD):
                return path

            if(q[0] < (current_observed_state.shape[0]-1) and current_observed_state[q[0]+1, q[1]] != WALL):
                if((q[0]+1, q[1]) not in explored):
                    Q.append(((q[0]+1, q[1]), path+["down"]))
            if(q[1] < (current_observed_state.shape[0]-1) and current_observed_state[q[0], q[1]+1] != WALL):
                if((q[0], q[1]+1) not in explored):
                    Q.append(((q[0], q[1]+1), path+["right"]))
            if(q[0] > 0 and current_observed_state[q[0]-1, q[1]] != WALL):
                if((q[0]-1, q[1]) not in explored):
                    Q.append(((q[0]-1, q[1]), path+["up"]))
            if(q[1] > 0 and current_observed_state[q[0], q[1]-1] != WALL):
                if((q[0], q[1]-1) not in explored):
                    Q.append(((q[0], q[1]-1), path+["left"]))

        return None

    def get_actions(self, current_observed_state, current_observation, previous_reward, run_infos):
        instance, action = self.get_in_memory(current_observation)
        if(previous_reward is not None and previous_reward > 0):
            # Copy k steps of previous map to current map
            for k in range(self.reward_steps_k):
                unused_index = max(0, len(self.unused_memory_action_map)-(k+1))
                self.memory_action_map.append(self.unused_memory_action_map[unused_index])

        if(instance is not None):
            self.unused_memory_action_map.append((current_observation, action))
            return [action], {}
        else:
            path = self.find_nearest_unobserved(current_observed_state)
            if(path is not None):

                random_action = path[0]
            else:
                random_action = random.choice(self.actions)

            self.unused_memory_action_map.append((current_observation, random_action))
            lenmap = len(self.unused_memory_action_map)
            if(lenmap > self.reward_steps_k):
                self.unused_memory_action_map = self.unused_memory_action_map[lenmap-1-self.reward_steps_k:lenmap-1]
            return [random_action], {}


class POMCP(Agent):
    def __init__(self,
                 stims,
                 agent_name="pomcp_simple",
                 search_depth=5,
                 tree_queries=100,
                 discount_factor=0.95,
                 **kwargs):

        print("kwargs: ", kwargs)
        self.alternate_policy=False
        self.agent_name = agent_name
        self.search_depth = search_depth
        self.tree_queries = tree_queries
        self.discount_factor = discount_factor
        self.stims = stims
        self.q=0

        # TODO: Remove this assumption
        assert len(list(set([stim.shape for stim in self.stims]))) == 1

        super(POMCP, self).__init__(stims, **kwargs)

    def get_actions(self, current_observed_state, current_observation, previous_reward, run_infos):
        run_info_extras = {}
        if(self.alternate_policy):
            return self.get_alternate_actions(current_observed_state, current_observation, previous_reward, run_infos), run_info_extras
        else:    

            if(self.agent_name == "pomcp_simple"):
                simple_reward_maps = fill_reward(current_observed_state, current_observed_state)
                simple_reward_map = None
                if(len(simple_reward_maps) == 0):
                    simple_reward_map = np.zeros(current_observed_state.shape)
                else:
                    simple_reward_map = simple_reward_maps[0]

                f=plt.figure()
                plt.imshow(simple_reward_map)
                f.savefig("./temp/rew{}.pdf".format(self.q), bbox_inches='tight') 
                pred_actions, policy = j.step_pomcp(current_observed_state,
                                                    self.search_depth,
                                                    self.observation_mode,
                                                    self.tree_queries,
                                                    self.discount_factor,
                                                    self.optimism, 
                                                    self.optimism*run_infos[-1]['obs_reward_map']+simple_reward_map,
                                                    run_infos[-1]['obs_reward_key'],
                                                    self.room_keys,
                                                    self.room_key_masks)
                run_info_extras['policy'] = self.extract_tree_stats(policy)
            
            elif(self.agent_name == "pomcp_ssp" or self.agent_name == "pomcp_mle"):
                maps, beliefs, reward_maps = run_infos[-1]['maps'], run_infos[-1]['beliefs'], run_infos[-1]['reward_maps']
                if(len(maps) > 0 and self.agent_name == "pomcp_mle"):
                    # Remove all but most likely
                    max_belief = beliefs[0]
                    max_map = maps[0]
                    max_reward_map = reward_maps[0]
                    for nmap, belief, reward_map in zip(maps, beliefs, reward_maps):
                        if(belief > max_belief):
                            max_belief = belief
                            max_map = nmap
                            max_reward_map = reward_map

                    f=plt.figure()
                    plt.imshow(max_map)
                    f.savefig("./temp/maxhyp{}.pdf".format(self.q), bbox_inches='tight')

                    maps = [max_map]
                    beliefs = [1.0]
                    reward_maps = [max_reward_map+[None, None]]+[None, None]


                pred_actions, valid_hyps, policy = j.step_pomcp_ssp(current_observed_state, 
                                                                    self.search_depth, 
                                                                    self.observation_mode, 
                                                                    self.tree_queries, 
                                                                    self.discount_factor, 
                                                                    self.optimism,
                                                                    maps, 
                                                                    beliefs, 
                                                                    reward_maps,
                                                                    self.optimism*run_infos[-1]['obs_reward_map'],
                                                                    run_infos[-1]['obs_reward_key'], 
                                                                    self.room_keys,
                                                                    self.room_key_masks)
                run_info_extras['valid_hyps'] = valid_hyps
                run_info_extras['policy'] = self.extract_tree_stats(policy)
            else:
                raise NotImplementedError
            return pred_actions, run_info_extras



