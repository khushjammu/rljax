import argparse
import os
from datetime import datetime

# from rljax.algorithm import SAC_Discrete
from rljax.env import make_atari_env
# from rljax.trainer import Trainer


# def run(args):
#     env = make_atari_env(args.env_id, sign_rewards=False, clip_rewards=True)
#     env_test = make_atari_env(args.env_id, episode_life=False, sign_rewards=False)

#     algo = SAC_Discrete(
#         num_agent_steps=args.num_agent_steps,
#         state_space=env.observation_space,
#         action_space=env.action_space,
#         seed=args.seed,
#     )

#     time = datetime.now().strftime("%Y%m%d-%H%M")
#     log_dir = os.path.join("logs", args.env_id, f"{str(algo)}-seed{args.seed}-{time}")

#     trainer = Trainer(
#         env=env,
#         env_test=env_test,
#         algo=algo,
#         log_dir=log_dir,
#         num_agent_steps=args.num_agent_steps,
#         action_repeat=4,
#         eval_interval=args.eval_interval,
#         seed=args.seed,
#     )
#     trainer.train()


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--env_id", type=str, default="MsPacmanNoFrameskip-v4")
#     p.add_argument("--num_agent_steps", type=int, default=3 * 10 ** 5)
#     p.add_argument("--eval_interval", type=int, default=5000)
#     p.add_argument("--seed", type=int, default=0)
#     args = p.parse_args()
#     run(args)

# import gym

from rljax.algorithm import FQF
from rljax.trainer import Trainer

NUM_AGENT_STEPS = 300000 # 20000
EVAL_INTERVAL = 50000 # 5000 # 1000
SEED = 0

# env = gym.make("CartPole-v0")
# env_test = gym.make("CartPole-v0")

env = make_atari_env("BreakoutNoFrameskip-v4", sign_rewards=False, clip_rewards=True)
env_test = make_atari_env("BreakoutNoFrameskip-v4", episode_life=False, sign_rewards=False)

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

# algo = FQF(
#         num_agent_steps=NUM_AGENT_STEPS,
#         state_space=env.observation_space,
#         action_space=env.action_space,
#         seed=SEED,
#         max_grad_norm=None,
#         gamma=0.99,
#         nstep=1,
#         buffer_size=10 ** 6,
#         use_per=False,
#         batch_size=32,
#         start_steps=50000,
#         update_interval=4,
#         update_interval_target=10000, # changed to fit pytorch one
#         eps=0.01,
#         eps_eval=0.001,
#         eps_decay_steps=250000,
#         loss_type="huber",
#         dueling_net=False,
#         double_q=False,
#         setup_net=True,
#         fn=None,
#         lr=5e-5,
#         lr_cum_p=2.5e-9,
#         units=(512,),
#         num_quantiles=32,
#         num_cosines=64,
#     )

algo = FQF(
        num_agent_steps=NUM_AGENT_STEPS,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=SEED,
        update_interval_target=10000, # changed to fit pytorch one
    )

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    log_dir="/tmp/rljax/fqf",
    num_agent_steps=NUM_AGENT_STEPS,
    eval_interval=EVAL_INTERVAL,
    seed=SEED,
)
trainer.train()
