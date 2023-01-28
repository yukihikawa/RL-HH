from gym.envs.registration import register

register(
    id = 'hh_env_ss-v0',
    entry_point = 'env.hh_env_ss:hh_env_ss', # 环境的入口
)