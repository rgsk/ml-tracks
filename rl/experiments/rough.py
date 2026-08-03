import numpy as np
import gymnasium as gym
DIMS = ["x (cart pos)", "x_dot (cart vel)", "theta (pole angle)", "theta_dot (pole vel)"]
BOX = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.2095, 0.2095], [-3.0, 3.0]])

def rollout_states(num_episodes=200, seed=0):
    env = gym.make('CartPole-v1')
    rng = np.random.default_rng(seed)
    states, deltas = [], []
    for ep in range(num_episodes):
        s, _ = env.reset(seed = seed + ep)
        done = False
        while not done:
            states.append(s)
            action = int(rng.integers(2))
            ns, _r, term, trunc, _ = env.step(action)
            deltas.append(np.abs(ns - s))
            done = term or trunc
            s = ns
        states.append(s)
    return np.array(states), np.array(deltas)

def to_cell(states, bins):
    lo, hi = BOX[:, 0], BOX[:, 1]
    idx = ((states - lo) / (hi - lo) * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    # each state as a 4-digit number written in base bins
    # id = i0*bins³ + i1*bins² + i2*bins¹ + i3
    # [1,0,2,1] -> 34
    result = np.ravel_multi_index(idx.T, (bins,) * 4)
    return result

def exp_table_dies(num_episodes=200, seed=0):
    states, deltas = rollout_states(num_episodes, seed)
    print(f'{states.shape=}, {deltas.shape=}')
    T = len(states)
    print(f'{num_episodes} random-policy episodes, {T:,} states visited')

    uniq = len(np.unique(states, axis=0))
    print('A')
    print(f'distinct raw states: {uniq:,} / {T:,}')
    print(f'repeats: {T - uniq}')
    print('B')
    env = gym.make('CartPole-v1')
    space = env.observation_space
    print(f'{'dim':22s} | observed min | observed max | env bounds')

    for i, name in enumerate(DIMS):
        bound = f'[{space.low[i]:.2f}, {space.high[i]:.2f}]'
        print(
            f"{name:22s} | {states[:, i].min():^+12.3f} | {states[:, i].max():^+12.3f}"
            f" | {bound}"
        )
    print('C')
    for bins in (3, 5, 10, 20, 50):
        cells = bins ** 4
        c = to_cell(states, bins)
        counts = np.bincount(c)
        seen = int((counts > 0).sum())
        print(f'{bins:5d}')
        
        
    print('Rough')
    print('hello')
    print('hello world')
    print('fuck me')
   

def run_experiments():
    exp_table_dies()

if __name__ == "__main__":
    run_experiments()
