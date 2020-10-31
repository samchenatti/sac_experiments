import argparse
import json
import subprocess
from datetime import datetime
from os import mkdir, path

from mysac.batch.numpy_batch import NumpySampledBuffer
from mysac.envs.pyrep_env import CartPoleEnv
from mysac.envs.stockenv import StockEnv
from mysac.models.mlp import PolicyModel, QModel
from mysac.sac.sac import SACAgent
from mysac.trainers.generic_train import generic_train


def run_experiment_from_specs(experiment_folder: str):
    """ Run an experiment from a dictionaty of specifications

    Args:
        experiment_folder: the path to the experiment folder. If does not
            exists, it is created. If it already exists, we try to load a file
            with specs within it. """

    # Create folders for saving experiment stats
    mkdir(experiment_folder)
    mkdir(experiment_folder + '/models/')
    mkdir(experiment_folder + '/stats/')

    meta = {
        'branch': subprocess.check_output(
            ["git", "branch"]).decode(),
        'commit': subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode(),
        'date': datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    }

    with open(experiment_folder + '/meta.json', 'w') as meta_file:
        json.dump(meta, meta_file)

    # env = CartPoleEnv(headless=False)
    env = StockEnv()

    buffer = NumpySampledBuffer(
        size=int(1e6), observation_size=6, action_size=1)

    agent = SACAgent(
        # Env
        env=env,

        # Models
        policy_model=PolicyModel(6, 1, 512),
        q1_model=QModel(6, 1, 512),
        q2_model=QModel(6, 1, 512),
        q1_target=QModel(6, 1, 512),
        q2_target=QModel(6, 1, 512),

        # Hyperparams
        gamma=0.99,
        policy_lr=3e-4,
        q_lr=3e-4,
        alpha_lr=3e-4,
        tau=5e-3
    )

    generic_train(
        env=env,
        agent=agent,
        buffer=buffer,
        experiment_folder=experiment_folder,
        batch_size=256,
        max_steps_per_episode=28800,
        sampled_steps_per_epoch=28800,
        train_steps_per_epoch=256
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sac for CartPole2d')

    parser.add_argument('--exp_path', type=str,
                        help='Output path for model binaries and stats')
    parser.add_argument('--deterministic', action='store_true',
                        help='Run the deterministic algorithm')
    parser.add_argument('--viz', action='store_true')

    args = parser.parse_args()

    if not args.deterministic:
        run_experiment_from_specs(args.exp_path)
