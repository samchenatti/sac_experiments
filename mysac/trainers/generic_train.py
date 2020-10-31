from typing import Callable

import numpy as np
from gym import Env
from mysac.samplers.sampler import BasicTrajectorySampler
from tqdm import tqdm
import torch


def generic_train(
        env: Env,
        agent: Callable,
        buffer: Callable,
        experiment_folder: str,
        batch_size: int = 256,
        max_steps_per_episode: int = 250,
        sampled_steps_per_epoch: int = 1000,
        train_steps_per_epoch: int = 1000):
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
        env.valid()
        trajectory = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=5000000,
            total_steps=50000000,
            deterministic=True,
            single_episode=True
        )

        print('Eval reward:', np.sum(trajectory['rewards']))
        with open(experiment_folder + '/stats/eval_stats.csv', 'a') as stat_f:
            stat_f.write(f'{np.sum(trajectory["rewards"])}\n')

        # Sample
        env.train()
        trajectory = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=max_steps_per_episode,
            total_steps=sampled_steps_per_epoch,
            single_episode=True
        )
        print('End sampling, adding to buffer...')

        buffer.add_trajectory(**trajectory)
        print('End buffer')

        # Train
        for _ in tqdm(range(train_steps_per_epoch)):
            batch = buffer.sample(batch_size)
            agent.train_from_samples(batch=batch)

        torch.save(agent.policy.state_dict(),
                   experiment_folder + '/models/policy.pt')
