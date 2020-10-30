from functools import partial
from typing import List

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.algorithm.sac import SAC
from rljax.network import ContinuousQuantileFunction, StateDependentGaussianPolicy
from rljax.util import quantile_loss


class TQC(SAC):
    name = "TQC"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        buffer_size=10 ** 6,
        use_per=False,
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(512, 512, 512),
        log_std_min=-20.0,
        log_std_max=2.0,
        d2rl=False,
        num_critics=5,
        num_quantiles=25,
        num_quantiles_to_drop=0,
    ):
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:

            def fn_critic(s, a):
                return ContinuousQuantileFunction(
                    num_critics=num_critics,
                    hidden_units=units_critic,
                    num_quantiles=num_quantiles,
                    d2rl=d2rl,
                )(s, a)

        if fn_actor is None:

            def fn_actor(s):
                return StateDependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    d2rl=d2rl,
                )(s)

        super(TQC, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
        )
        cum_p = jnp.arange(0, num_quantiles + 1, dtype=jnp.float32) / num_quantiles
        self.cum_p_prime = jnp.expand_dims((cum_p[1:] + cum_p[:-1]) / 2.0, 0)
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles
        self.num_quantiles_target = num_quantiles * num_critics - num_quantiles_to_drop

    @partial(jax.jit, static_argnums=0)
    def _calculate_q(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> jnp.ndarray:
        return jnp.concatenate(self._calculate_q_list(params_critic, state, action), axis=1)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
        next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        next_quantile = self._calculate_q(params_critic_target, next_state, next_action)
        next_quantile = jnp.sort(next_quantile)[:, : self.num_quantiles_target]
        next_quantile -= jnp.exp(log_alpha) * self._calculate_log_pi(next_action, next_log_pi)
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_quantile)

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
        self,
        quantile_list: List[jnp.ndarray],
        target: jnp.ndarray,
        weight: np.ndarray,
    ) -> jnp.ndarray:
        loss_critic = 0.0
        for quantile in quantile_list:
            loss_critic += quantile_loss(target[:, None, :] - quantile[:, :, None], self.cum_p_prime, weight, "huber")
        loss_critic /= self.num_critics * self.num_quantiles
        abs_td = jnp.abs(target[:, None, :] - quantile_list[0][:, :, None]).mean(axis=1).mean(axis=1, keepdims=True)
        return loss_critic, abs_td
