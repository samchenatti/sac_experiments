
import numpy as np
from gym.envs.registration import register
from src.config import Config
from src.env.dataset.stock import StockDataset, StockMarketInfo
from src.env.environment import StockEnvironment


class StockEnv(StockEnvironment):
    def __init__(self, *args, **kwargs):
        config = Config.get_config(filename="/home/samuel/Develop/taperead/"
                                   "taperead3/config.INI")

        stock_market_info = StockMarketInfo(config=config)
        stock_market_info.load()

        dataset = StockDataset(
            env_id=0,
            config=config,
            stock_market_info=stock_market_info
        )
        dataset.set_up_stock_dataset()

        super().__init__(config=config, env_id=0, dataset=dataset)

        self._max_episode_steps = 28800

    def reset(self):
        return np.array(super().reset()['state'][0]).astype('float32')

    def step(self, action):
        if action >= -1 and action <= -0.3:
            action = 0

        elif action > -0.3 and action < 0.35:
            action = 1

        else:
            action = 2

        obs, reward, done, info = super().step(action)

        if done:
            obs = -1 * np.ones((6,))

        else:
            obs = np.array(obs['state'][0]).astype('float32')

        return obs, reward, done, info


register(
    id='stock-v0',
    entry_point='stockenv:StockEnv',
)
