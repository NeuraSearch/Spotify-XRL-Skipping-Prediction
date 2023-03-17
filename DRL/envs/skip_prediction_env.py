import enum
import gym
import gym.spaces
import numpy as np

# import useful constants from lib/constants.py
from lib import constants as consts

# action space: NoSkip and Skip
class Actions(enum.Enum):
    NoSkip = 0
    Skip = 1

# create the environment using the gym class definition
class SkipPredictionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # class constructor. Parameters:
    # df: the train dataframe
    # df_Y: the target variable (the skip attribute to predict)
    # is_train: a flag to indicate whether we are performing training or evaluation
    def __init__(self, df, df_Y, is_train, obs_stacking, no_temporal_leakage):
        super(SkipPredictionEnv, self).__init__()

        self.df = df
        self.df_Y = df_Y

        # df_size stores how many records we have in the train set
        self.df_size = self.df.shape[0]
        self.is_train = is_train
        self.obs_stacking = obs_stacking
        self.no_temporal_leakage = no_temporal_leakage

        # index value to perform indexing on the dataframes to access data at ith position
        self.current_step = 0

        self.action_space = gym.spaces.Discrete(n=len(Actions))

        if self.no_temporal_leakage:
            self.n_variables = consts.STATE_N_VARS
            self.__state_vars = consts.STATE_VARIABLES
        else:
            self.n_variables = consts.TEMPORAL_LEAKAGE_STATE_N_VARS
            self.__state_vars = consts.TEMPORAL_LEAKAGE_STATE_VARIABLES

        if self.obs_stacking:
            self.n_history_states = 4
            observation_space_dim = self.n_variables * self.n_history_states
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_space_dim,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_variables,), dtype=np.float32)

    # return the next observation that is given to the RL agent. This is "fetched" from the dataframe (self.df) at a certain index (self.current_step)
    def __next_observation(self):
        if self.obs_stacking:
            obs = np.zeros(shape=self.observation_space.shape)
            end = self.n_history_states * self.n_variables
            update_offset = self.n_variables
            start = end - update_offset

            for his_idx in range(self.n_history_states):
                offset = self.current_step - his_idx
                obs[start:end] = self.df[offset]

                # if reached beginning of episode, stop
                curr_session_position = self.df[offset, self.__state_vars.SessionPosition.value]
                if curr_session_position == 1:
                    break

                end = start
                start -= update_offset
        else:
            obs = np.empty(shape=self.observation_space.shape)
            obs = self.df[self.current_step]

        return obs

    # perform a step given an action
    def step(self, action):
        # from the integer value of the action, get the name (e.g., Actions.Skip)
        action_name = Actions(action)

        # obtain the truth value from the target. Index refers to the "skip_2" feature
        # in this work we predict this feature
        curr_session_position = self.df[self.current_step, self.__state_vars.SessionPosition.value]
        if self.no_temporal_leakage:
            curr_skip = self.df_Y[self.current_step][2]
            curr_session_length = self.df_Y[self.current_step][0]
        else:
            curr_skip = self.df_Y[self.current_step][0]
            curr_session_length = self.df[self.current_step, self.__state_vars.SessionLength.value]

        # set `done` flag if reached end of episode (session_position == session_length)
        done = curr_session_position == curr_session_length

        # terminate episode on wrong action. This is done only during training.
        # if correct action is sampled, then +1 reward
        reward = 0
        if action_name == Actions.Skip:
            if curr_skip:
                reward += 1
            elif self.is_train:
                done = True
        elif action_name == Actions.NoSkip:
            if not curr_skip:
                reward += 1
            elif self.is_train:
                done = True

        # if termination was reached due to wrong action, update `current_step` to be at the
        #  end of current episode. Then normal update of +1 is done as usual
        if done and self.is_train:
            self.current_step = int(self.current_step + (curr_session_length - curr_session_position))

        # go to next record --> +1 on the self.current_step index
        self.current_step += 1

        # start from beginning (index 0) if reached end of dataframe
        if self.current_step > self.df_size - 1:
            self.current_step = 0

        # fetch the next observation
        obs = self.__next_observation()

        return obs, reward, done, {}

    def reset(self):
        return self.__next_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass
