from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from net_env import NetworkEnv
import networkx as nx

g = nx.read_gml('nsfnet.gml')
def env_creator(env_config):
	return NetworkEnv(g)
register_env("netenv-v0", env_creator)

config = PPOConfig().training(vf_clip_param=20.0)
config = config.rollouts(num_rollout_workers=1)
#config = config.training(gamma=0.999, lr=0.001)
algo = config.build(env="netenv-v0")

#config = config.training(gamma=0.999, lr=0.0)
#baseline = config.build(env="netenv-v1")

for _ in range(20):
	algo.train()
	#baseline.train()