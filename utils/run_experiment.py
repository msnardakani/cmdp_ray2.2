



def run_experiment():
    agent_config_sp = [{"target_mean": target_means[0]},
                       {'curriculum': 'self_paced',
                        "target_mean": target_means[1], "target_var": target_vars[1],
                        "init_mean": init_mean, "init_var": init_var},
                       {'curriculum': 'self_paced',
                        "target_mean": target_means[2], "target_var": target_vars[2],
                        "init_mean": init_mean, "init_var": init_var}]

    agent_config_eval = [{"target_mean": target_means[0]},
                         {"target_mean": target_means[1], },
                         {"target_mean": target_means[2], }]

    agent_config_default = [{"target_mean": target_means[0], },
                            {"target_mean": target_means[1], "target_var": target_vars[2], },
                            {"target_mean": target_means[2], "target_var": target_vars[2], }]

    if target_vars[0] is not None:
        agent_config_sp[0]['target_var'] = target_vars[0]
        agent_config_eval[0]['target_var'] = target_vars[0]
        agent_config_default[0]['target_var'] = target_vars[0]

    if target_prior is not None:
        agent_config_sp[0]['target_priors'] = target_prior
        agent_config_eval[0]['target_priors'] = target_prior
        agent_config_default[0]['target_priors'] = target_prior

    dummy_env = SPEnv(config={})
    model_config = {"fcnet_hiddens": model,
                    "fcnet_activation": "relu",
                    }
    dist_class, logit_dim = ModelCatalog.get_action_dist(
        dummy_env.action_space, model_config, framework='torch'
    )

    central_policy, _ = FC_MLP(
        obs_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        num_outputs=logit_dim,
        model_config=model_config,
    )

    def gen_policy(i):
        out = dict()
        for i in range(i):
            if i == 0:
                config = {
                    "custom_model": 'central_model',
                    "custom_model_config": {"distilled_model": central_policy, },
                    "q_model_config": model_config,
                    "policy_model_config": model_config,

                }
                out['distilled_policy'] = PolicySpec(config=config)
            else:
                config = {
                    "custom_model": 'local_model',
                    "custom_model_config": {"distilled_model": central_policy, },
                    "q_model_config": model_config,
                    "policy_model_config": model_config,
                    # # "gamma": random.choice([0.95, 0.99]),
                    "num_steps_sampled_before_learning_starts": 1000

                }
                out["learner_{}".format(i - 1)] = PolicySpec(config=config)

        return out

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == 0:
            return "distilled_policy"
        else:
            return "learner_{}".format(agent_id - 1)

    def policy_mapping_fn_dummy(agent_id, episode, worker, **kwargs):
        if agent_id == 0:
            return "distilled_policy"
        else:
            return "learner_0"

    # Setup PPO with an ensemble of `num_policies` different policies.
    env_config = {"num_agents": num_agents, "agent_config": agent_config_sp}

    policies = gen_policy(num_agents)
    policy_ids = list(policies.keys())
    multiagent_config = {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": list(policies.keys())[1:],
        "count_steps_by": "agent_steps",
    }
    policies_dummy = gen_policy(2)
    policy_ids_dummy = list(policies_dummy.keys())
    dummy_multiagent_config = {
        'policies': policies_dummy,
        "policy_mapping_fn": policy_mapping_fn_dummy,
        "count_steps_by": "agent_steps",
        "policies_to_train": policy_ids_dummy[1:],
    }