from net_env_copy import NetworkEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms import dqn
import networkx as nx

g = nx.read_gml('nsfnet.gml')
def env_creator(env_config):
    return NetworkEnv(g)

register_env("netenv-v0", env_creator)

config = DQNConfig()


replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 60000,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-06,
    }


config = config.training(replay_buffer_config=replay_config)
config = config.rollouts(num_rollout_workers=2)
config = config.resources(num_gpus=0)
config = config.env_runners(num_env_runners=1)
config = config.environment("netenv-v0")

algo = dqn.DQN(config=config)

for _ in range(40):
	algo.train()

'''
save_result = algo.save()
path_to_checkpoint = save_result.checkpoint.path
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{path_to_checkpoint}'."
)
'''