from ray.rllib.policy.policy import PolicySpec


def dummy_policy_mapping_fn(agent_id, episode, worker, **kwargs):

    return list(worker.config.policies.keys())[0]


def policy_mapping_fn(agent_id, episode, worker, **kwargs):

    return list(worker.config.policies.keys())[agent_id]


def gen_ppo_distral_policy(N, obs_space, model_config, central_policy, central_policy_target =None, ctx_mode=0, sample_offset = 500):

    ctx_aug = obs_space[0] if ctx_mode == 2 else None
    if central_policy_target is not None:
        return {"learner_{}".format(i): PolicySpec(config={'model': {
            "custom_model": 'local',
            "custom_model_config": {'central':{"custom_model": 'local',
                                               'custom_model_config':{"distilled_model": central_policy,  'model': model_config, 'ctx_aug': ctx_aug},
                                               },
                                    'target_central':{'custom_model': 'local',
                                                      'custom_model_config':{"distilled_model": central_policy_target,  'model': model_config, 'ctx_aug': ctx_aug},}
        },},
            "num_steps_sampled_before_learning_starts": sample_offset

                                                         }) for i in range(N)}
    else:
        return {"learner_{}".format(i): PolicySpec(config={'model': {
            "custom_model": 'local',
            "custom_model_config": {"distilled_model": central_policy,  'model': model_config, 'ctx_aug': ctx_aug},
        },
            "num_steps_sampled_before_learning_starts": sample_offset

                                                         }) for i in range(N)}


#
# def gen_sac_distral_policy(N, obs_space, model_config, central_policy, ctx_mode=0, sample_offset = 500):
#
#         return {"learner_{}".format(i): PolicySpec(config={ "custom_model": 'local_model',
#                     "custom_model_config": {"distilled_model": central_policy, },
#                     "q_model_config": model_config,
#                     "policy_model_config": model_config,
#                     "num_steps_sampled_before_learning_starts": sample_offset
#
#         }) for i in range(N)}



