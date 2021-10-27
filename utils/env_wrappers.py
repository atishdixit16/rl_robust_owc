import gym
from stable_baselines3.common.type_aliases import GymStepReturn
import numpy as np
from gym import spaces
from collections import deque

class StepReset(gym.Wrapper):
    def __init__(self, env: gym.Env, steps_max: int = 1):
        """
        'steps_max' no. of steps with reset

        """
        gym.Wrapper.__init__(self, env)
        self.steps_max = steps_max
        self.reset_reward = 0.0

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        self.reset_reward = 0.0
        for _ in range(self.steps_max):
            action = np.array([1]*self.env.action_space.shape[0])
            obs, rew, done, _ = self.env.step(action)
            self.reset_reward = self.reset_reward + rew
        return obs
    
    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)            
        if self.env.episode_step == (self.steps_max+1):
            reward = reward + self.reset_reward
            
        return state, reward, done, info
    
    
class StateCoarse(gym.Wrapper):
    def __init__(self, env: gym.Env, x_coords: np.ndarray, y_coords: np.ndarray, include_well_pr: bool = False):
        """
        'x_coords' : x coordinates of the state grid to be considered
        'y_coords' : y coordinates of the state grid to be considered
        'include_well_pr' : append well (injectors and producers) pressure to states (assume zero pressure in reset)
        """
        gym.Wrapper.__init__(self, env)
        self.include_well_pr = include_well_pr
        self.x_coords = x_coords
        self.y_coords = y_coords
        if self.include_well_pr:
            high = np.array([1e5]* (x_coords.shape[0] + env.n_inj + env.n_prod) )
        else:
            high = np.array([1e5]*x_coords.shape[0])
        obs_space = spaces.Box(low= -high, high=high, dtype=np.float64)
        self.observation_space = obs_space

    def reset(self, **kwargs) -> np.ndarray:
        state = self.env.reset(**kwargs)
        state = state.reshape((self.env.grid.nx, self.env.grid.ny))
        obs = []
        for x,y in zip(self.x_coords, self.y_coords):
            obs.append(state[x,y])
        if self.include_well_pr:
            ps = np.zeros(self.env.n_inj + self.env.n_prod)
            obs = np.append(obs, ps)              
        return np.array(obs)
    
    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)            
        state = state.reshape((self.env.grid.nx, self.env.grid.ny))
        obs = []
        for x,y in zip(self.x_coords, self.y_coords):
            obs.append(state[x,y])
        if self.include_well_pr:
            ps = []
            p_scaled = np.interp(self.env.solverP.p, (self.env.solverP.p.min(), self.env.solverP.p.max()), (-1,1))
            for x,y in zip(self.env.i_x, self.env.i_y):
                ps.append(p_scaled[x,y])
            for x,y in zip(self.env.p_x, self.env.p_y):
                ps.append(p_scaled[x,y])
            obs = np.append(obs, ps)   
        return np.array(obs), reward, done, info
    
    
class BufferWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_steps: int):
        """
        n_steps: number of steps to stack in state
        
        """
        gym.Wrapper.__init__(self, env)
        self.state_queue = deque(maxlen=n_steps)
        self.n_steps = n_steps
        high_ = self.env.observation_space.high.repeat(n_steps, axis=0)
        low_ = self.env.observation_space.low.repeat(n_steps, axis=0)
        obs_space = spaces.Box(low=low_, high=high_, dtype=np.float64)
        self.observation_space = obs_space

    def reset(self, **kwargs) -> np.ndarray:
        state = self.env.reset(**kwargs)
        for _ in range(self.n_steps):
            self.state_queue.append(state)
        return np.array(self.state_queue).reshape(-1)
    
    def step(self, action) -> GymStepReturn:
        state, reward, done, info = self.env.step(action)            
        self.state_queue.append(state)
        return np.array(self.state_queue).reshape(-1), reward, done, info