from gym.envs.registration import register

register(
    id='hh_env-v0',
    entry_point='env.hh_env:hh_env',  # 环境的入口
)
