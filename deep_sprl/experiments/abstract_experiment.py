import os
import time
import pickle
import numpy as np
import torch
from abc import ABC, abstractmethod
from enum import Enum

from imitation.algorithms import bc

from deep_sprl.util.vec_normalize import VecNormalize
from deep_sprl.util.parameter_parser import create_override_appendix

# from deep_sprl.util.funcs import DnCCallback


from stable_baselines3.sac import SAC
from stable_baselines3 import PPO as PPO2
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.gail.dataset.dataset import ExpertDataset
from stable_baselines3.sac.policies import MlpPolicy as SACMlpPolicy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CurriculumType(Enum):
    GoalGAN = 1
    ALPGMM = 2
    SelfPaced = 3
    Default = 4
    Random = 5
    SelfPacedv2 = 6

    def __str__(self):
        if self.goal_gan():
            return "goal_gan"
        elif self.alp_gmm():
            return "alp_gmm"
        elif self.self_paced():
            return "self_paced"
        elif self.self_paced_v2():
            return "self_paced_v2"
        elif self.default():
            return "default"
        else:
            return "random"

    def self_paced(self):
        return self.value == CurriculumType.SelfPaced.value

    def self_paced_v2(self):
        return self.value == CurriculumType.SelfPacedv2.value

    def goal_gan(self):
        return self.value == CurriculumType.GoalGAN.value

    def alp_gmm(self):
        return self.value == CurriculumType.ALPGMM.value

    def default(self):
        return self.value == CurriculumType.Default.value

    def random(self):
        return self.value == CurriculumType.Random.value

    @staticmethod
    def from_string(string):
        if string == str(CurriculumType.GoalGAN):
            return CurriculumType.GoalGAN
        elif string == str(CurriculumType.ALPGMM):
            return CurriculumType.ALPGMM
        elif string == str(CurriculumType.SelfPaced):
            return CurriculumType.SelfPaced
        elif string == str(CurriculumType.SelfPacedv2):
            return CurriculumType.SelfPacedv2
        elif string == str(CurriculumType.Default):
            return CurriculumType.Default
        elif string == str(CurriculumType.Random):
            return CurriculumType.Random
        else:
            raise RuntimeError("Invalid string: '" + string + "'")


class AgentInterface(ABC):

    def __init__(self, learner, obs_dim):
        self.learner = learner
        self.obs_dim = obs_dim

    def estimate_value(self, inputs):
        if isinstance(self.learner.env, VecNormalize):
            return self.estimate_value_internal(self.learner.env.normalize_obs(inputs))
        else:
            return self.estimate_value_internal(inputs)

    @abstractmethod
    def estimate_value_internal(self, inputs):
        pass

    @abstractmethod
    def mean_policy_std(self, cb_args, cb_kwargs):
        pass

    def save(self, log_dir, ext = ""):
        self.learner.save(os.path.join(log_dir, "model"+ext))
        if isinstance(self.learner.env, VecNormalize):
            self.learner.env.save(os.path.join(log_dir, "normalizer"+ext+".pkl"))


class SACInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value_internal(self, inputs):
        inputs = torch.Tensor(inputs).to('cuda')
        # return np.squeeze(self.learner.sess.run([self.learner.step_ops[6]], {self.learner.observations_ph: inputs}))
        next_actions, next_log_prob = self.learner.actor.action_log_prob(inputs)
        # Compute the next Q values: min over all critics targets
        next_q_values = torch.cat(self.learner.critic_target(inputs, next_actions), dim=1)

        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        return np.squeeze(next_q_values.cpu().detach().numpy())


    def mean_policy_std(self, cb_args, cb_kwargs):
        if "infos_values" in cb_args[0] and len(cb_args[0]["infos_values"]) > 0:
            return cb_args[0]["infos_values"][4]
        else:
            return np.nan


class TRPOInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value_internal(self, inputs):

        return self.learner.policy.predict_values(torch.from_numpy(inputs).to(self.learner.device)).cpu()

    def mean_policy_std(self, cb_args, cb_kwargs):
        # log_std = np.squeeze(self.learner.sess.run([self.learner.policy_pi.proba_distribution.logstd],
                                                   # {self.learner.policy_pi.obs_ph: np.zeros((1, self.obs_dim))})[0])
        # return np.mean(np.exp(log_std))
        return np.nan


class PPOInterface(AgentInterface):

    def __init__(self, learner, obs_dim):
        super().__init__(learner, obs_dim)

    def estimate_value_internal(self, inputs):
        return self.learner.policy.predict_values(torch.from_numpy(inputs).to(self.learner.device)).cpu()

    def mean_policy_std(self, cb_args, cb_kwargs):
        # log_std = np.squeeze(self.learner.sess.run([self.learner.train_model.proba_distribution.logstd],
                                                   # {self.learner.train_model.obs_ph: np.zeros((1, self.obs_dim))})[0])
        # return np.mean(np.exp(log_std))
        return np.nan


class MAPPOInterface(AgentInterface):

    def __init__(self, learner, obs_dim, idx=0):
        super().__init__(learner, obs_dim)
        self.idx =idx

    def estimate_value_internal(self, inputs):
        return self.learner.policy.predict_values(torch.from_numpy(inputs).to(self.learner.device)).cpu()

    def mean_policy_std(self, cb_args, cb_kwargs):
        # log_std = np.squeeze(self.learner.sess.run([self.learner.train_model.proba_distribution.logstd],
        # {self.learner.train_model.obs_ph: np.zeros((1, self.obs_dim))})[0])
        # return np.mean(np.exp(log_std))
        return np.nan
    def save(self, log_dir):
        self.learner.save(os.path.join(log_dir, "model_"+str(self.idx)))
        if isinstance(self.learner.env, VecNormalize):
            self.learner.env.save(os.path.join(log_dir, "normalizer_"+str(self.idx)+".pkl"))

class SACEvalWrapper:

    def __init__(self, model):
        self.model = model

    def step(self, observation, state=None, deterministic=False):
        return self.model.predict(observation, state=state, deterministic=deterministic)[0]


class PPOTRPOEvalWrapper:

    def __init__(self, model):
        self.model = model

    def step(self, observation, state=None, deterministic=False):
        if len(observation.shape) == 1:
            observation = observation[None, :]
            return self.model.predict(observation, state=state, deterministic=deterministic)[0][0, :]
        else:
            return self.model.predict(observation, state=state, deterministic=deterministic)[0]


class Learner(Enum):
    TRPO = 1
    PPO = 2
    SAC = 3
    DnC = 4
    def __str__(self):
        if self.trpo():
            return "trpo"
        elif self.ppo():
            return "ppo"
        elif self.dnc():
            return "dnc"
        else:
            return "sac"

    def trpo(self):
        return self.value == Learner.TRPO.value

    def ppo(self):
        return self.value == Learner.PPO.value

    def sac(self):
        return self.value == Learner.SAC.value
    def dnc(self):
        return self.value == Learner.DnC.value

    def create_learner(self, env, parameters):
        if (self.trpo() or self.ppo()) and not issubclass(type(env), VecEnv):
            env = DummyVecEnv([lambda: env])

        if self.trpo():
            model = TRPO('MlpPolicy', env, **parameters["common"], **parameters[str(self)])
            interface = TRPOInterface(model, env.observation_space.shape[0])
        elif self.ppo():
            model = PPO2('MlpPolicy', env, **parameters["common"], **parameters[str(self)])
            interface = PPOInterface(model, env.observation_space.shape[0])

        # elif self.dnc():
        #     model =
        else:
            model = SAC(SACMlpPolicy, env, **parameters["common"], **parameters[str(self)])
            interface = SACInterface(model, env.observation_space.shape[0])

        # if "pretrain_data_path" in parameters:
        #     data_path = parameters["pretrain_data_path"]
        #     model.pretrain(ExpertDataset(expert_path=data_path, verbose=0), n_epochs=25)

        return model, interface

    def load(self, path, env):
        if self.trpo():
            return TRPO.load(path, env=env)
        elif self.ppo():
            return PPO2.load(path, env=env)
        else:
            return SAC.load(path, env=env)

    def load_for_evaluation(self, path, env, dnc = False):
        if (self.trpo() or self.ppo()) and not issubclass(type(env), VecEnv):
            env = DummyVecEnv([lambda: env])
        if dnc:
            return PPOTRPOEvalWrapper(bc.reconstruct_policy(path))
        model = self.load(path, env)

        if self.sac():
            return SACEvalWrapper(model)
        else:
            return PPOTRPOEvalWrapper(model)

    @staticmethod
    def from_string(string):
        if string == str(Learner.TRPO):
            return Learner.TRPO
        elif string == str(Learner.PPO):
            return Learner.PPO
        elif string == str(Learner.SAC):
            return Learner.SAC
        elif string == str(Learner.DnC):
            return Learner.DnC
        else:
            raise RuntimeError("Invalid string: '" + string + "'")

class DnCCallback:

    def __init__(self, log_directory, learner, env_wrapper, sp_teacher=None, idx = 0,n_inner_steps=1, n_offset=0,
                 save_interval=5, step_divider=1 ,use_true_rew=False, cluster=None):
        self.log_dir = os.path.realpath(log_directory)
        self.learner = learner
        # print(learner)
        self.env_wrapper = env_wrapper
        self.sp_teacher = sp_teacher
        self.n_offset = n_offset
        self.n_inner_steps = n_inner_steps
        self.save_interval = save_interval
        self.algorithm_iteration = 0
        self.step_divider = step_divider
        self.iteration = 0
        self.last_time = None
        self.use_true_rew = use_true_rew
        self.idx = idx
        self.format = "   %4d    | %4d | %.1E |   %3d    |  %.2E  |  %.2E  |  %.2E   "
        if self.sp_teacher is not None:
            context_dim = self.sp_teacher.context_dist.mean().shape[0]
            text = "| [%.2E"
            for i in range(0, context_dim - 1):
                text += ", %.2E"
            text += "] "
            self.format += text + text

        header = " Iteration |  Agent  |  Time   | Ep. Len. | Mean Reward | Mean Disc. Reward | Mean Policy STD "
        if self.sp_teacher is not None:
            header += "|     Context mean     |      Context std     "
        self.cluster=None
        if self.idx==0:
            print(header)
            self.cluster = cluster


    def __call__(self, *args, **kwargs):
        if self.algorithm_iteration % self.step_divider == 0:
            data_tpl = (self.iteration,self.idx,)

            t_new = time.time()
            dt = np.nan
            if self.last_time is not None:
                dt = t_new - self.last_time
            data_tpl += (dt,)

            mean_rew, mean_disc_rew, mean_length = self.env_wrapper.get_statistics()
            data_tpl += (int(mean_length), mean_rew, mean_disc_rew)

            data_tpl += (self.learner.mean_policy_std(args, kwargs),)

            if self.sp_teacher is not None:
                if self.iteration >= self.n_offset and self.iteration % self.n_inner_steps == 0:
                    vf_inputs, contexts, rewards = self.env_wrapper.get_context_buffer()
                    self.sp_teacher.update_distribution(mean_disc_rew, contexts,
                                                        rewards if self.use_true_rew else self.learner.estimate_value(
                                                            vf_inputs))
                context_mean = self.sp_teacher.context_dist.mean()
                context_std = np.sqrt(np.diag(self.sp_teacher.context_dist.covariance_matrix()))
                data_tpl += tuple(context_mean.tolist())
                data_tpl += tuple(context_std.tolist())

            print(self.format % data_tpl)

            if self.iteration % self.save_interval == 0:
                iter_log_dir = os.path.join(self.log_dir, "iteration-" + str(self.iteration))
                os.makedirs(iter_log_dir, exist_ok=True)
                if self.idx == 0 :
                    # print('cluster_saved')
                    with open(os.path.join(iter_log_dir,'clusters.pkl'), 'wb') as handle:
                            pickle.dump(self.cluster, handle)


                self.learner.save(iter_log_dir)
                if self.sp_teacher is not None:
                    self.sp_teacher.save(os.path.join(iter_log_dir, "context_dist_"+str(self.idx)))

            self.last_time = time.time()
            self.iteration += 1

        self.algorithm_iteration += 1



class ExperimentCallback:

    def __init__(self, log_directory, learner, env_wrapper, sp_teacher=None, n_inner_steps=1, n_offset=0,
                 save_interval=5, step_divider=1, use_true_rew=False, print_header= True):
        self.log_dir = os.path.realpath(log_directory)
        self.learner = learner

        self.env_wrapper = env_wrapper
        self.sp_teacher = sp_teacher
        self.n_offset = n_offset
        self.n_inner_steps = n_inner_steps
        self.save_interval = save_interval
        self.algorithm_iteration = 0
        self.step_divider = step_divider
        self.iteration = 0
        self.last_time = None
        self.use_true_rew = use_true_rew

        self.format = "   %4d    | %.1E |   %3d    |  %.2E  |  %.2E  |  %.2E   "
        if self.sp_teacher is not None:
            context_dim = self.sp_teacher.context_dist.mean().shape[0]
            text = "| [%.2E"
            for i in range(0, context_dim - 1):
                text += ", %.2E"
            text += "] "
            self.format += text + text

        header = " Iteration |  Time   | Ep. Len. | Mean Reward | Mean Disc. Reward | Mean Policy STD "
        if self.sp_teacher is not None:
            header += "|     Context mean     |      Context std     "
        if print_header:
            print(header)

    def __call__(self, *args, **kwargs):
        if self.algorithm_iteration % self.step_divider == 0:
            data_tpl = (self.iteration,)

            t_new = time.time()
            dt = np.nan
            if self.last_time is not None:
                dt = t_new - self.last_time
            data_tpl += (dt,)

            mean_rew, mean_disc_rew, mean_length = self.env_wrapper.get_statistics()
            data_tpl += (int(mean_length), mean_rew, mean_disc_rew)

            data_tpl += (self.learner.mean_policy_std(args, kwargs),)

            if self.sp_teacher is not None:
                if self.iteration >= self.n_offset and self.iteration % self.n_inner_steps == 0:
                    vf_inputs, contexts, rewards = self.env_wrapper.get_context_buffer()
                    self.sp_teacher.update_distribution(mean_disc_rew, contexts,
                                                        rewards if self.use_true_rew else self.learner.estimate_value(
                                                            vf_inputs))
                context_mean = self.sp_teacher.context_dist.mean()
                context_std = np.sqrt(np.diag(self.sp_teacher.context_dist.covariance_matrix()))
                data_tpl += tuple(context_mean.tolist())
                data_tpl += tuple(context_std.tolist())

            print(self.format % data_tpl)

            if self.iteration % self.save_interval == 0:
                iter_log_dir = os.path.join(self.log_dir, "iteration-" + str(self.iteration))
                os.makedirs(iter_log_dir, exist_ok=True)

                self.learner.save(iter_log_dir)
                if self.sp_teacher is not None:
                    self.sp_teacher.save(os.path.join(iter_log_dir, "context_dist"))

            self.last_time = time.time()
            self.iteration += 1

        self.algorithm_iteration += 1


class AbstractExperiment(ABC):
    APPENDIX_KEYS = {"default": ["DISCOUNT_FACTOR", "STEPS_PER_ITER", "LAM"],
                     CurriculumType.SelfPaced: ["ALPHA_OFFSET", "MAX_KL", "OFFSET", "ZETA"],
                     CurriculumType.SelfPacedv2: ["PERF_LB", "MAX_KL", "OFFSET"],
                     CurriculumType.GoalGAN: ["GG_NOISE_LEVEL", "GG_FIT_RATE", "GG_P_OLD"],
                     CurriculumType.ALPGMM: ["AG_P_RAND", "AG_FIT_RATE", "AG_MAX_SIZE"],
                     CurriculumType.Random: [],
                     CurriculumType.Default: []}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, experiment_name = '', view=False, use_true_rew=False, ):
        self.base_log_dir = base_log_dir
        self.parameters = parameters
        self.curriculum = CurriculumType.from_string(curriculum_name)
        self.learner = Learner.from_string(learner_name)
        self.seed = seed
        self.view = view
        self.name = experiment_name
        self.use_true_rew = use_true_rew
        self.process_parameters()
        self.dnc=False
        self.N = 2
    @abstractmethod
    def create_experiment(self):
        pass

    @abstractmethod
    def get_env_name(self):
        pass

    @abstractmethod
    def create_self_paced_teacher(self):
        pass

    @abstractmethod
    def evaluate_learner(self, path):
        pass

    def get_other_appendix(self):
        return self.name

    @staticmethod
    def parse_max_size(val):
        if val == "None":
            return None
        else:
            return int(val)

    @staticmethod
    def parse_n_hidden(val):
        val = val.replace(" ", "")
        if not (val.startswith("[") and val.endswith("]")):
            raise RuntimeError("Invalid list specifier: " + str(val))
        else:
            vals = val[1:-1].split(",")
            res = []
            for v in vals:
                res.append(int(v))
            return res

    def process_parameters(self):
        allowed_overrides = {"DISCOUNT_FACTOR": float, "MAX_KL": float, "ZETA": float, "ALPHA_OFFSET": int,
                             "OFFSET": int, "STEPS_PER_ITER": int, "LAM": float, "AG_P_RAND": float, "AG_FIT_RATE": int,
                             "AG_MAX_SIZE": self.parse_max_size, "GG_NOISE_LEVEL": float, "GG_FIT_RATE": int,
                             "GG_P_OLD": float, "PERF_LB": float}
        for key in sorted(self.parameters.keys()):
            if key not in allowed_overrides:
                raise RuntimeError("Parameter '" + str(key) + "'not allowed'")

            value = self.parameters[key]
            tmp = getattr(self, key)
            if isinstance(tmp, dict):
                tmp[self.learner] = allowed_overrides[key](value)
            else:
                setattr(self, key, allowed_overrides[key](value))

    def get_log_dir(self):
        override_appendix = create_override_appendix(self.APPENDIX_KEYS["default"], self.parameters)
        leaner_string = str(self.learner)
        key_list = self.APPENDIX_KEYS[self.curriculum]
        # print(key_list)
        for key in sorted(key_list):
            tmp = getattr(self, key)
            if isinstance(tmp, dict):
                if self.learner.value==4:

                    tmp = tmp[Learner.PPO]
                else:
                    tmp = tmp[self.learner]

            leaner_string += "_" + key + "=" + str(tmp).replace(" ", "")

        if self.use_true_rew:
            leaner_string += "_TRUEREWARDS"
        if self.dnc:
            leaner_string += "_DNC_KL_pen="+str(self.PENALTY) +"_distill_interval="+str(self.distillation_period) + "_bc_itr="+str(self.bc_iteration)+"_bc_samples="+str(self.bc_samples)
        return os.path.join(self.base_log_dir, self.get_env_name(), str(self.curriculum),
                            leaner_string,   self.get_other_appendix()+ override_appendix, "seed-" + str(self.seed))

    def train(self):
        model, timesteps, callback_params = self.create_experiment()
        log_directory = self.get_log_dir()


        # print(log_directory)

        if os.path.exists(os.path.join(log_directory, "performance.pkl")):
            print("Log directory already exists! Going directly to evaluation")
        else:
            if self.dnc:
                callback = [DnCCallback(log_directory=log_directory, **cb) for cb in callback_params]
                model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=callback)
            else:
                callback_params["use_true_rew"] = self.use_true_rew
                callback = ExperimentCallback(log_directory=log_directory, **callback_params)
                model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=callback)

    def evaluate(self):
        log_dir = self.get_log_dir()

        iteration_dirs = [d for d in os.listdir(log_dir) if d.startswith("iteration-")]
        unsorted_iterations = np.array([int(d[len("iteration-"):]) for d in iteration_dirs])
        idxs = np.argsort(unsorted_iterations)
        sorted_iteration_dirs = np.array(iteration_dirs)[idxs].tolist()
        # print(sorted_iteration_dirs)
        # First evaluate the KL-Divergences if Self-Paced learning was used
        if (self.curriculum.self_paced() or self.curriculum.self_paced_v2()) and not \
                os.path.exists(os.path.join(log_dir, "kl_divergences.pkl")) and not self.dnc:
            kl_divergences = []
            for iteration_dir in sorted_iteration_dirs:
                teacher = self.create_self_paced_teacher()
                iteration_log_dir = os.path.join(log_dir, iteration_dir)
                teacher.load(os.path.join(iteration_log_dir, "context_dist.npy"))
                kl_divergences.append(teacher.target_context_kl())

            kl_divergences = np.array(kl_divergences)
            with open(os.path.join(log_dir, "kl_divergences.pkl"), "wb") as f:
                pickle.dump(kl_divergences, f)
        # print(log_dir)
        if self.dnc:
            for i in range(self.N+1):
                if not os.path.exists(os.path.join(log_dir, "performance_"+str(i)+".pkl")):
                    # print('elav')
                    seed_rewards = []
                    seed_disc_rews = []
                    for iteration_dir in sorted_iteration_dirs:
                        iteration_log_dir = os.path.join(log_dir, iteration_dir)
                        perf = self.evaluate_learner(iteration_log_dir)
                        print("Agent: "+str(i)+"Evaluated " + iteration_dir + ": mean reward: " + str(
                            np.mean(perf[0])) + ', mean discounted reward: ' + str(np.mean(perf[1])))
                        seed_rewards.append(perf[0])
                        seed_disc_rews.append(perf[1])

                    seed_rewards = np.array(seed_rewards)
                    seed_disc_rews = np.array(seed_disc_rews)

                    with open(os.path.join(log_dir,  "performance_"+str(i)+".pkl"), "wb") as f:
                        pickle.dump((seed_rewards, seed_disc_rews), f)


        else:

            if not os.path.exists(os.path.join(log_dir, "performance.pkl")):
                # print('elav')
                seed_rewards = []
                seed_disc_rews = []
                for iteration_dir in sorted_iteration_dirs:
                    iteration_log_dir = os.path.join(log_dir, iteration_dir)
                    perf = self.evaluate_learner(iteration_log_dir)
                    print("Evaluated " + iteration_dir + ": mean reward: " + str(np.mean(perf[0]))+ ', mean discounted reward: '+ str(np.mean(perf[1])) )
                    seed_rewards.append(perf[0])
                    seed_disc_rews.append(perf[1])

                seed_rewards = np.array(seed_rewards)
                seed_disc_rews = np.array(seed_disc_rews)

                with open(os.path.join(log_dir, "performance.pkl"), "wb") as f:
                    pickle.dump((seed_rewards, seed_disc_rews), f)
