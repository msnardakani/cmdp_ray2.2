
from torch.distributions import MultivariateNormal
import numpy as np
import torch

def kl_div(p, q):
    epsilon = 1e-10

    # You may want to instead make copies to avoid changing the np arrays.


    return np.sum(p * np.log((p+epsilon) / (q+epsilon)))

class CategoricalSelfPacedTeacher:

    def __init__(self, target_probs,  init_probs,  perf_lb, max_diff):


        self.max_diff = max_diff
        self.update_ctr = 0
        self.context_N = target_probs.shape[0]
        # self.context_bounds = context_bounds
        # self.bounds = context_bounds
        # self.use_avg_performance = use_avg_performance
        self.perf_lb = perf_lb
        self.perf_lb_reached = False
        self.vals = np.array(range(self.context_N))

        self.probs = init_probs
        self.target_probs = target_probs


    def target_context_kl(self, numpy=True):

        return kl_div( self.probs, self.target_probs)

    def save(self, path):

        np.save(path, self.probs)

    def load(self, path):
        self.probs = np.load(path)


    def get_task(self):
        return self.probs

    def set_task(self, task):
        self.probs = task


    def _compute_context_kl(self, old_context_dist):
        return kl_div(self.probs, old_context_dist)

    def _compute_expected_performance(self, probs, cons_t, old_probs, c_val_t):
        con_ratio_t = probs[cons_t] /old_probs+ 1e-4
        return torch.mean(con_ratio_t * c_val_t)

    def get_context(self):
        return {"probs":self.probs,  "kl_div": self.target_context_kl()}

    def get_status(self):
        return {"mean_diff": np.mean(np.abs(self.probs - self.target_probs)),
                'perf_lb': self.perf_lb_reached}

    def export_dist(self):

        return dict(probs=self.probs)

    def update_distribution(self, contexts, values, ):
        V_bar = np.array([np.nan_to_num(np.nanmean(values, where=contexts==i)) for i in self.vals])
        V_bar0 = np.mean(V_bar)
        V_bar = V_bar - V_bar0
        V_bar_norm = np.linalg.norm(V_bar)
        V_ = self.perf_lb - V_bar0
        if np.dot(self.probs, V_bar) >= V_:
            # print("Optimizing KL")

            self.perf_lb_reached = True
            gamma3 = 0
            denum_term = self.max_diff*V_bar_norm**2 - (V_  - np.dot(self.probs, V_bar))**2
            if denum_term>0:
                num_term = np.linalg.norm(self.target_probs - self.probs)**2 - (np.dot(self.target_probs - self.probs, V_bar))**2
                gamma3 = -1 + (num_term/ denum_term)**0.5

            gamma3 = gamma3 if gamma3>0 else 0

            gamma2 = 2*((V_ - np.dot(self.target_probs, V_bar)) + gamma3* (V_ - np.dot(self.probs, V_bar)))/V_bar_norm**2
            probs =self.probs + (1/(2 + 2*gamma3))*(2*(self.target_probs - self.probs) + gamma2*V_bar)

            probs = np.clip(probs, a_min=0, a_max = 1)
            self.probs = probs/np.sum(probs)



            return


        elif not self.perf_lb_reached:
            probs = np.clip(self.probs+ (self.max_diff**.5*V_bar)/np.linalg.norm(V_bar), a_min=0, a_max=1)
            self.probs = probs/np.sum(probs)

            return



    def sample(self):
        sample = np.random.choice(self.vals, p=self.probs)
        return sample




# class DiscreteGaussianSelfPacedTeacher(GaussianSelfPacedTeacher):
#
#     def __init__(self, target_mean, initial_mean, context_bounds, perf_lb, init_covar_scale = 0.01,
#                  max_kl=0.1, std_lower_bound=None, kl_threshold=None):
#
#
#         target_variance = np.eye(target_mean.shape[0])*0.01
#         super(DiscreteGaussianSelfPacedTeacher, self).__init__( target_mean, target_variance, initial_mean, context_bounds, perf_lb, init_covar_scale =init_covar_scale,
#              max_kl=max_kl, std_lower_bound=std_lower_bound, kl_threshold=kl_threshold)
#         # self.ctx_conversion_fn = self._clip_round_ctx
#
#
#     def sample(self):
#         sample = self.context_dist.rsample().detach().numpy()
#         return np.clip(np.round(sample), self.context_bounds[0], self.context_bounds[1])
