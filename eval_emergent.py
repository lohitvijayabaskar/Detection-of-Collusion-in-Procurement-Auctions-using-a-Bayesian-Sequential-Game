import os
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from emergent_procurement_env import (
    ProcurementEmergentEnv,
    calibrate_emergent_likelihoods,
    calibrate_emergent_thresholds,
)
from train_emergent_rl import (
    env_creator, RESERVE_PRICE, TOTAL_FIRMS, ROUNDS_PER_EPISODE, CHECKPOINT_DIR,
)


def evaluate_model():
    ray.init(ignore_reinit_error=True)

    env_name = "emergent_procurement_env"
    register_env(env_name, env_creator)

    print("Re-calibrating regulator (same procedure as training, independent seed)...")
    l0, l1 = calibrate_emergent_likelihoods(
        reserve_price=RESERVE_PRICE, total_firms=TOTAL_FIRMS,
        n_episodes=150, rounds_per_episode=ROUNDS_PER_EPISODE, seed0=99,
    )
    cusum_h, sr_h = calibrate_emergent_thresholds(
        l0, l1, reserve_price=RESERVE_PRICE, total_firms=TOTAL_FIRMS,
        n_episodes=100, rounds_per_episode=ROUNDS_PER_EPISODE, seed0=20_099,
    )

    env_config = {"likelihood_h0": l0, "likelihood_h1": l1,
                  "cusum_threshold": cusum_h, "sr_threshold": sr_h}

    temp_env = ProcurementEmergentEnv(likelihood_h0=l0, likelihood_h1=l1)
    obs_space = temp_env.observation_space("firm01")
    act_space = temp_env.action_space("firm01")

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "shared_policy"

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name, env_config=env_config)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(num_env_runners=1, explore=False)
        .resources(num_gpus=0)
    )
    algo = config.build_algo()

    print(f"\n---> Loading trained model from {CHECKPOINT_DIR}...\n")
    algo.restore(CHECKPOINT_DIR)

    env = ProcurementEmergentEnv(
        reserve_price=RESERVE_PRICE, total_firms=TOTAL_FIRMS,
        rounds_per_episode=ROUNDS_PER_EPISODE,
        likelihood_h0=l0, likelihood_h1=l1,
        cusum_threshold=cusum_h, sr_threshold=sr_h,
    )
    obs, _ = env.reset(seed=42)

    print("=" * 90)
    print(" EVALUATION: CONVERGED FIRM POLICY vs. CLOSED-LOOP BAYESIAN-SEQUENTIAL REGULATOR")
    print("=" * 90)

    for r in range(1, ROUNDS_PER_EPISODE + 1):
        actions = {
            a: algo.compute_single_action(observation=obs[a], policy_id="shared_policy", explore=False)
            for a in env.agents
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        g = infos.pop("__common__")

        sorted_agents = sorted(infos.items(), key=lambda kv: kv[1]["bid"])
        print(f"\n--- ROUND {r:2d} --- P(cartel)={g['posterior_collusion']:.3f} "
              f"alarm={'YES' if g['alarm'] else 'no'}")
        for idx, (agent, info) in enumerate(sorted_agents[:5]):
            flag = " <== WINNER" if agent == g["winner"] else ""
            strat = info.get("strategy", "?")
            print(f"    {idx+1:02d}. {agent} (Strat {strat}) | markup={info['markup']:.3f}x | "
                  f"bid=${info['bid']:.2f} | reward=${info['reward']:.2f}{flag}")

        if all(terms.values()):
            break

    ray.shutdown()


if __name__ == "__main__":
    evaluate_model()