from gym.envs.registration import register


register(
    id="PlayTableSimEnv",
    entry_point="calvin_env.envs.play_table_env:PlayTableSimEnv",
)

register(
	id="TouchCube",
	entry_point="robomimic.envs.env_flat_cube:TouchCubeImageEnv",
)
