import numpy as np
import pandas as pd
import gym
import math
from sarsa import Sarsa


def main(
    n_iters=3000,
    n_games_per_update=10,
    n_max_steps=1000,
    n_bins = 50):

    env = gym.make('CartPole-v0')
    model = Sarsa(actions=range(2), alpha=0.1, gamma=0.95, epsilon=0.5)

    cart_p_bins = pd.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]
    cart_v_bins = pd.cut([-2.0, 2.0], bins=n_bins, retbins=True)[1][1:-1]
    pole_a_bins = pd.cut([-math.radians(41.8), math.radians(41.8)], bins=n_bins, retbins=True)[1][1:-1]
    pole_v_bins = pd.cut([-3.0, 3.0], bins=n_bins, retbins=True)[1][1:-1]
    bins = (cart_p_bins, cart_v_bins, pole_a_bins, pole_v_bins)

    # training
    for iter in range(n_iters):
        finished_steps = []
        for game in range(n_games_per_update):
            obs = env.reset()
            state = build_state(obs, bins)
            action = model.choose_action(state)
            for step in range(n_max_steps):
                obs, reward, done, info = env.step(action)
                next_state = build_state(obs, bins)
                next_action = model.choose_action(next_state)
                model.update_q(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                if done:
                    finished_steps.append(step)
                    break
        print("[%d / %d]: %.1f" % (iter, n_iters, (sum(finished_steps)/len(finished_steps)) ))    

    # testing
    obs = env.reset()
    state = build_state(obs, bins)
    done = False
    count = 0
    while not done:
        env.render()
        action = model.choose_action(state, training=False)
        obs, reward, done, info = env.step(action)
        state = build_state(obs, bins)
        count += 1
    print(count)


def build_state(obs, bins):
    cart_p, cart_v, pole_a, pole_v = obs
    cart_p_bins, cart_v_bins, pole_a_bins, pole_v_bins = bins
    features = [to_bin(cart_p, cart_p_bins), to_bin(cart_v, cart_v_bins),
                to_bin(pole_a, pole_a_bins), to_bin(pole_v, pole_v_bins)]
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


if __name__ == '__main__':
    main()