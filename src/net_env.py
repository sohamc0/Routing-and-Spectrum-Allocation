import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box
import numpy as np
import networkx as nx
import random

class NetworkEnv(gym.Env):

    def __init__(self, g: nx.Graph) -> None:
        # define state space
        self.g = g
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()

        # creating a node name (string) to id (int) dictionary
        count = 0
        self._node_word_to_num = {}
        for n in list(g.nodes):
            self._node_word_to_num[n] = count
            count += 1

        # creating a edge (string tuple) to id (int) dictionary
        count = 0
        self._edge_to_num = {}
        for e in list(g.edges):
            self._edge_to_num[e] = count
            count += 1
        
        self.observation_space = Dict(
            {
                # represents the 10 links for each of the edges in the graph
                "links": Box(0, 20, shape=(num_edges,10), dtype=int),
                # represents [source, target, holding_time]
                "req": Box(low=np.array([0,0,10]), high=np.array([num_nodes-1, num_nodes-1, 20]), shape=(3,), dtype=int)
            }
        )

        # each request will have at most 3 different paths to select. If none can fulfill, then a blocking action is taken
        self.action_space = Discrete(4)

        self.round = 0

        # creating a dictionary which will hold 3 disjoint paths for each (source, target) node combination
        self.paths = {}
        for s in g.nodes:
            for t in g.nodes:
                # making sure that (t, s) is not already in dictionary. If it is, then no point in adding (s, t)
                if s != t and (self._node_word_to_num[t], self._node_word_to_num[s]) not in self.paths:
                    s_node_num = self._node_word_to_num[s]
                    t_node_num = self._node_word_to_num[t]
                    self.paths[(s_node_num, t_node_num)] = [] # this list will eventually have 3 paths
                    all_possible_paths = list(nx.edge_disjoint_paths(g, s, t, cutoff=3))
                    # for now, we are choosing the 3 shortest paths
                    for i in range(3):
                        if i >= len(all_possible_paths): 
                            # in case there are less than 3 disjoint paths, we will just duplicate the first
                            # path to fill in the missing paths
                            self.paths[(s_node_num, t_node_num)].append(all_possible_paths[0])
                        else:
                            self.paths[(s_node_num, t_node_num)].append(all_possible_paths[i])

                    # because the path list contains nodes instead of edges, we have to convert this
                    # to edges by some preprocessing done below...
                    prev_paths = self.paths[(s_node_num, t_node_num)]
                    self.paths[(s_node_num, t_node_num)] = []
                    for i in range(3):
                        p = prev_paths[i]
                        new_path = []
                        for node_ind in range(len(p) - 1):
                            a = p[node_ind]
                            b = p[node_ind + 1]

                            # converting (a, b) from their "string-tuple" format
                            # to their edge_id (int) by using the dict, _edge_to_num
                            a_b = (a, b)
                            if (a, b) not in self._edge_to_num:
                                a_b = (b, a)

                            new_path.append(self._edge_to_num[a_b])
                        self.paths[(s_node_num, t_node_num)].append(new_path)

    def _generate_req(self):
        # a request (indicating the capacity required to host this request)
        min_ht = 10
        max_ht = 20
        ht = np.random.randint(min_ht, max_ht)

        
        # following 5 lines is for case 2 only
        s = random.choice(list(self.g.nodes))
        t = random.choice(list(self.g.nodes))
        # we don't want s and t to be equal
        while s == t:
            t = random.choice(list(self.g.nodes))
        

        '''
        # following 2 lines is for case 1 only
        s = 'San Diego Supercomputer Center'
        t = 'Jon Von Neumann Center, Princeton, NJ'
        '''

        return np.array([self._node_word_to_num[s], self._node_word_to_num[t], ht])
    
    def _get_obs(self):
        return {
            "links": self._linkstates,
            "req": self._req
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # resetting all used links to 0
        self._linkstates = np.array([[0] * 10] * self.g.number_of_edges())

        # to generate a request 
        self._req = self._generate_req()

        observation = self._get_obs()
        info = {}

        self.round = 0
        self.blocks = 0

        return observation, info

    def step(self, action):
        # decrementing all holding times
        for e in self._linkstates:
            for i in range(len(e)):
                if e[i] > 0:
                    e[i] -= 1

        self.round += 1
        terminated = (self.round == 100) # True if it experienced 100 rounds

        # action = 0 (P0), 1 (P1), 2 (P2), 3
        blocking_action = 3

        s = self._req[0]
        t = self._req[1]
        ht = self._req[2]

        # first, orienting (s,t) correctly...
        s_t = (s, t)
        if s_t not in self.paths:
            s_t = (t, s)

        if action != blocking_action:
            # the following list will contain the number of in-use links for every
            # edge in the chosen path IF the path were to be chosen (we want <= 10 for every edge)
            num_used_slots = []

            # for each edge, e, on the chosen path...
            for e in self.paths[s_t][action]:
                # now finding how many colors are already used on this edge (i.e. holding time for link is > 0)
                curr_used = 0
                for i in range(len(self._linkstates[e])):
                    if self._linkstates[e][i] > 0:
                        curr_used += 1
                num_used_slots.append(curr_used + 1) # adding 1 because of the new request being accomodated
        if action != blocking_action and all(ele <= 10 for ele in num_used_slots):
            # the action chosen is OK
            # updating 1 color on each edge of chosen path with holding time
            for e in self.paths[s_t][action]:
                # updating 1 color
                colors_ls = self._linkstates[e]
                for i in range(len(colors_ls)):
                    if self._linkstates[e][i] == 0:
                        colors_ls[i] = ht
                        self._linkstates[e] = colors_ls
                        break

            reward = +1 * ht
        else: # we need to block
            # No update on the state
            self.blocks += 1
            reward = -1

        self._req = self._generate_req()
        observation = self._get_obs()
        info = {}
        
        return observation, reward, terminated, False, info