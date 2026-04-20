from gym.envs.registration import register


register(
	id="TouchCube",
	entry_point="robomimic.envs.env_flat_cube:TouchCubeImageEnv",
)
