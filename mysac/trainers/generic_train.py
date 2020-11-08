from typing import Callable, Dict, List

import numpy as np
import torch
from gym import Env
from mysac.samplers.sampler import BasicTrajectorySampler
from tqdm import tqdm

from .utils import save_sac_models


def generic_train(
        env: Env,
        agent: Callable,
        buffer: Callable,
        experiment_folder: str,
        batch_size: int = 256,
        max_steps_per_episode: int = 250,
        sampled_steps_per_epoch: int = 1000,
        train_steps_per_epoch: int = 1000,
        evaluator: Callable[[Dict[str, torch.tensor]], None] = None):
    """ Generic, infinite train loop with deterministic evaluation.

    Args:
        agent: the agent interface (should expose a get_action interface)
        buffer: the data structure to keep the replay buffer
        experiment_folder: path to the experiment folder where the models and
            evaluation data will be saved
        batch_size: the size of batch that will be sampled from the replay
            buffer at each train step
        max_steps_per_episode: the max steps per episode during sample
        sampled_steps_per_epoch: how many steps should be sampled at each
            epoch
        train_steps_per_epoch: how many backward passes per epoch
    """
    while True:
        # Eval
        trajectory = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=500,
            total_steps=500,
            deterministic=True
        )

        print('Eval reward:', np.sum(trajectory['rewards'])/500)
        with open(experiment_folder + '/stats/eval_stats.csv', 'a') as stat_f:
            stat_f.write(f'{np.sum(trajectory["rewards"])}\n')

        # Sample
        trajectory = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=max_steps_per_episode,
            total_steps=sampled_steps_per_epoch
        )

        buffer.add_trajectory(**trajectory)

        for _ in tqdm(range(train_steps_per_epoch)):
            batch = buffer.sample(batch_size)

            agent_info: Dict[str, torch.tensor] = agent \
                .train_from_samples(batch=batch)

            if evaluator:
                evaluator.aggregate_values(**agent_info)

        save_sac_models(agent=agent, experiment_folder=experiment_folder)

        if evaluator:
            evaluator.save_metrics()
