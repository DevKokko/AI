from gym.envs.registration import register

register(
    id='NineMensMorris-v0',
    entry_point='ninemensmorris.envs:NineMensMorrisEnv',
)
