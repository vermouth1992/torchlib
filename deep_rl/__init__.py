from gym.envs.registration import register
from torchlib.utils.random.torch_random_utils import set_global_seeds
import numpy as np

register(
    id='CartPoleContinuous-v0',
    entry_point='torchlib.deep_rl.envs:CartPoleContinuous',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPoleContinuous-v1',
    entry_point='torchlib.deep_rl.envs:CartPoleContinuous',
    max_episode_steps=500,
    reward_threshold=475.0,
)


class BaseAgent(object):
    def predict(self, state):
        raise NotImplementedError


def test(env, agent: BaseAgent, num_episode=100, frame_history_len=1, render=False, seed=1996):
    set_global_seeds(seed)
    env.seed(seed)
    reward_lst = []
    for i in range(num_episode):
        observation_lst = []
        done = False
        episode_reward = 0
        previous_observation = env.reset()
        observation_lst.append(previous_observation)
        for _ in range(frame_history_len - 1):
            if render:
                env.render()
            action = env.action_space.sample()
            previous_observation, reward, done, _ = env.step(action)
            observation_lst.append(previous_observation)
            episode_reward += reward
        while not done:
            if render:
                env.render()
            action = agent.predict(np.concatenate(observation_lst, axis=-1))
            previous_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            observation_lst.pop(0)
            observation_lst.append(previous_observation)
        print('Episode: {}. Reward: {}'.format(i, episode_reward))
        reward_lst.append(episode_reward)
    print('Reward range [{}, {}]'.format(np.min(reward_lst), np.max(reward_lst)))
    print('Reward {}±{}'.format(np.mean(reward_lst), np.std(reward_lst)))

    env.close()
