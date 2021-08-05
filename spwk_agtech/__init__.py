from gym.envs.registration import register

register(
    id="PCSE-v0",
    entry_point="spwk_agtech.pcse_env:PCSE_Env",
)