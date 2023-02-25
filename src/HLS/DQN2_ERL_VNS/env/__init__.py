from gym.envs.registration import register

register(
    id='vns_env-v0',
    entry_point='env.vns_env:vns_env',  # 环境的入口
)
register(
    id='vns_env2-v0',
    entry_point='env.vns_env2:vns_env2',  # 环境的入口
)
