from gym.envs.registration import register

register(
    id='skip_prediction-v0',
    entry_point='DRL.envs:SkipPredictionEnv',
)
