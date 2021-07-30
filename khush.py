import gym

from rljax.algorithm import DQN
from rljax.trainer import Trainer

NUM_AGENT_STEPS = 20000
SEED = 0

# env = gym.make("CartPole-v0")
# env_test = gym.make("CartPole-v0")

env = gym.make("BreakoutNoFrameskip-v4")
env_test = gym.make("BreakoutNoFrameskip-v4")

# algo = DQN(
#     num_agent_steps=NUM_AGENT_STEPS,
#     state_space=env.observation_space,
#     action_space=env.action_space,
#     seed=SEED,
#     batch_size=256,
#     start_steps=1000,
#     update_interval=1,
#     update_interval_target=400,
#     eps_decay_steps=0,
#     loss_type="l2",
#     lr=1e-3,
# )

algo = FQF(
    num_agent_steps=100000,
    state_space=env.observation_space,
    action_space=env.action_space,
    seed=SEED,
    batch_size=256,
    start_steps=1000,
    update_interval=1,
    update_interval_target=400,
    eps_decay_steps=0,
    loss_type="l2",
    lr=1e-3,
    use_per=True,
    dueling_net=True,
    double_q=True,
)

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    log_dir="/tmp/rljax/dqn",
    num_agent_steps=NUM_AGENT_STEPS,
    eval_interval=1000,
    seed=SEED,
)
trainer.train()
