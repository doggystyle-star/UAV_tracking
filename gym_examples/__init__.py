from gym.envs.registration import register
register(
    id="gym_examples/GridWorld-v0",
    entry_point="gym_examples.envs:GridWorldEnv",
    max_episode_steps=300,
)
register(
    id="gym_examples/GazeboWorld-v0",
    entry_point="gym_examples.envs:GazeboWorldEnv",
    max_episode_steps=300,
)
register(
    id="gym_examples/EGazeboWorld-v0",
    entry_point="gym_examples.envs:EGazeboWorldEnv",
    max_episode_steps=4000
)

