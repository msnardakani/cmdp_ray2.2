
from torch.distributions import MultivariateNormal
import numpy as np
import torch



def w_project_norm(x, W, y):
    return np.matmul(np.matmul(x,W),y)
class GaussianSelfPacedTeacher:

    def __init__(self,target_mean, target_variance, initial_mean, context_bounds, perf_lb, init_covar_scale = 0.01,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, use_avg_performance=False):


        self.max_kl = max_kl
        self.update_ctr = 0
        self.context_dim = target_mean.shape[0]
        self.context_bounds = context_bounds
        self.bounds = context_bounds
        self.use_avg_performance = use_avg_performance
        self.perf_lb = perf_lb
        self.perf_lb_reached = False
        if std_lower_bound is not None and kl_threshold is None:
            raise RuntimeError("Error! Both Lower Bound on standard deviation and kl threshold need to be set")
        else:
            if std_lower_bound is not None:
                if isinstance(std_lower_bound, np.ndarray):
                    if std_lower_bound.shape[0] != self.context_dim:
                        raise RuntimeError("Error! Wrong dimension of the standard deviation lower bound")
                elif std_lower_bound is not None:
                    std_lower_bound = np.ones(self.context_dim) * std_lower_bound
            self.std_lower_bound = std_lower_bound
            self.kl_threshold = kl_threshold

        # Create the initial context distribution

        # self.scale = init_covar_scale
        # Create the target distribution
        if isinstance(target_variance, np.ndarray):
            target_covar = target_variance
        else:
            target_covar = target_variance * np.eye(self.context_dim)



        self.ctx_mean = initial_mean
        self.target_mean = target_mean
        self.target_covar = target_covar
        self.target_cov_inv = np.linalg.inv(target_covar)
        self.ctx_covar_scale = init_covar_scale
        self.covar_scale_lb = np.mean(self.std_lower_bound)
        self.covar_scale_ub = 1/self.covar_scale_lb

        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor((1 / self.ctx_covar_scale) * self.target_covar, dtype=torch.float64))
        self.target_dist = MultivariateNormal(loc=torch.as_tensor(self.target_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor( self.target_covar, dtype=torch.float64))

    def target_context_kl(self, numpy=True):
        kl_div = torch.distributions.kl.kl_divergence(self.context_dist,
                                                      self.target_dist).detach()
        if numpy:
            kl_div = kl_div.numpy()

        return kl_div

    def save(self, path):
        weights = np.concatenate([self.ctx_mean, self.ctx_covar_scale])
        np.save(path, weights)

    def load(self, path):
        weights = np.load(path)
        self.ctx_mean = weights[0:-1]
        self.ctx_covar_scale = weights[-1]
        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor((1 / self.ctx_covar_scale) * self.target_covar, dtype=torch.float64))

    def get_task(self):
        return np.concatenate([self.ctx_mean, self.ctx_covar_scale])

    def set_task(self, task):
        self.ctx_mean = task[0:-1]
        self.ctx_covar_scale = task[-1]
        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor((1 / self.ctx_covar_scale) * self.target_covar, dtype=torch.float64))

    def _compute_context_kl(self, old_context_dist):
        return torch.distributions.kl.kl_divergence(self.context_dist, old_context_dist)

    def _compute_expected_performance(self, dist, cons_t, old_c_log_prob_t, c_val_t):
        con_ratio_t = torch.exp(dist.log_prob(cons_t) - old_c_log_prob_t)
        return torch.mean(con_ratio_t * c_val_t)

    def get_context(self):
        return {"mean":self.ctx_mean, "var": np.diag((1 / self.ctx_covar_scale) * self.target_covar), "kl_div": self.target_context_kl()}

    def get_status(self):
        return {"mean_diff": np.mean(np.abs(self.ctx_mean - self.target_mean)),
                "var_diff": np.mean(
                    np.abs(np.diag((1 / self.ctx_covar_scale) * self.target_covar- self.target_covar))),
                'perf_lb': self.perf_lb_reached}

    def export_dist(self):

        return dict(target_mean=[self.ctx_mean,],
                    target_var=[(1 / self.ctx_covar_scale) * self.target_covar,],
                    target_priors=np.array([1,]))

    def update_mean(self, avg_performance, contexts, values, ):
        V_bar = np.mean(values)
        self.update_ctr += 1
        if V_bar >= self.perf_lb:
            # print("Optimizing KL")
            self.perf_lb_reached = True
            u_bar = -np.dot( values, contexts)/len(values)
            u_bar_norm = w_project_norm(u_bar, self.target_cov_inv, u_bar)
            du_norm = w_project_norm(self.target_mean - self.ctx_mean, self.target_cov_inv, self.target_mean - self.ctx_mean)
            du_u_bar_norm = w_project_norm(self.target_mean - self.ctx_mean, self.target_cov_inv,
                                     u_bar)


            term1 = (V_bar - self.perf_lb)**2 - 2*self.max_kl*self.ctx_covar_scale*u_bar_norm
            term2 = du_norm *u_bar_norm - du_u_bar_norm** 2

            # if len(term1)>1:
            #     print(term1)
            #     print(u_bar, u_bar_norm)
            #     print(du_norm, du_u_bar_norm )
            #     print(V_bar)

            if term1>0:
                gamma2 = self.ctx_covar_scale * u_bar_norm ** 0.5 * (term2 / term1) ** 0.5
                if (1 + gamma2* self.ctx_covar_scale)*(V_bar-self.perf_lb) - self.ctx_covar_scale*du_u_bar_norm >0:

                    ctx_mean =self.ctx_mean + (1/(self.ctx_covar_scale* u_bar_norm**0.5))*(term1/term2)**0.5*(
                            self.target_mean - self.ctx_mean + (1/u_bar_norm)*((u_bar_norm**0.5 *term2*(V_bar- self.perf_lb)/term1)-
                                                                           du_u_bar_norm)* u_bar)
                else:
                    # (V_bar - self.perf_lb)**2 - self.ctx_covar_scale*du_u_bar_norm>0:
                    ctx_mean = self.ctx_mean +(2*self.max_kl/(self.ctx_covar_scale*du_u_bar_norm))**0.5 * (self.target_mean - self.ctx_mean)
            elif (V_bar - self.perf_lb) - self.ctx_covar_scale* du_u_bar_norm>0:
                ctx_mean = self.target_mean + (( (V_bar - self.perf_lb) - self.ctx_covar_scale* du_u_bar_norm)/(self.ctx_covar_scale * u_bar_norm) )* u_bar
            else:
                ctx_mean = self.target_mean
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])

            return


        elif not self.perf_lb_reached:
            u_bar = np.dot( values, contexts)/len(values)

            ctx_mean = self.ctx_mean + (2*self.max_kl/(np.matmul(np.matmul(u_bar,self.target_cov_inv),u_bar)*self.ctx_covar_scale))**.5 *u_bar
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])
            return


    def update_covar_scale(self, avg_performance, contexts, values, ):
        V_bar = np.mean(values)
        u_bar = np.mean(values*np.sum(np.matmul(self.ctx_mean - contexts, self.target_cov_inv) * (self.ctx_mean - contexts), axis=1))
        self.update_ctr += 1
        if V_bar >= self.perf_lb:
            # print("Optimizing KL")
            self.perf_lb_reached = True

            if self.ctx_covar_scale <1:
                if u_bar*self.ctx_covar_scale - self.context_dim*V_bar>0:
                    scale = self.ctx_covar_scale + min(2*self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5,
                            (2*(V_bar- self.perf_lb) + u_bar*self.ctx_covar_scale)/(u_bar - V_bar*self.context_dim/self.ctx_covar_scale))
                else:
                    scale = self.ctx_covar_scale + 2*self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5
                if np.isnan(scale):
                    return
                self.ctx_covar_scale = np.clip(scale, a_min=self.ctx_covar_scale, a_max=1)
                return

            else:

                if u_bar*self.ctx_covar_scale - self.context_dim*V_bar<0:
                    scale = self.ctx_covar_scale + min(-2*self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5,
                            (2*(V_bar- self.perf_lb) + u_bar*self.ctx_covar_scale)/(u_bar - V_bar*self.context_dim/self.ctx_covar_scale))
                else:
                    scale = self.ctx_covar_scale -2*self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5
                if np.isnan(scale):
                    return
                self.ctx_covar_scale = np.clip(scale, a_min=1, a_max=self.ctx_covar_scale)
                return


        elif not self.perf_lb_reached:
            # u_bar = np.mean(
            #     values * np.sum(np.matmul(self.ctx_mean - contexts, self.target_cov_inv) * (self.ctx_mean - contexts),
            #                     axis=1))

            # u_bar = np.mean(np.sum(np.matmul(self.ctx_mean - contexts, self.target_cov_inv) * (self.ctx_mean - contexts), axis=1))
            # u_bar = np.mean(self.ctx_mean - contexts, axis=0)
            # np.matmul(np.matmul(u_bar, target_cov_inv), u_bar)
            s_ = 2*self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5
            s_new = self.ctx_covar_scale+ s_ if self.context_dim - u_bar*self.ctx_covar_scale > 0 else self.ctx_covar_scale - s_
            if np.isnan(s_new):
                return
            self.ctx_covar_scale = np.clip(s_new, a_min = self.covar_scale_lb, a_max = self.covar_scale_ub)
            return

    def update_distribution(self, avg_performance, contexts, values, ):
        if self.update_ctr%2 ==0:
            self.update_mean(avg_performance=avg_performance, contexts= contexts, values=values)
        else:
            self.update_covar_scale(avg_performance=avg_performance, contexts= contexts, values=values)

        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor((1 / self.ctx_covar_scale) * self.target_covar, dtype=torch.float64),
                                               )

    def sample(self):
        sample = self.context_dist.rsample().detach().numpy()
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])




class DiscreteGaussianSelfPacedTeacher(GaussianSelfPacedTeacher):

    def __init__(self, target_mean, initial_mean, context_bounds, perf_lb, init_covar_scale = 0.01,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None):


        target_variance = np.eye(target_mean.shape[0])*0.01
        super(DiscreteGaussianSelfPacedTeacher, self).__init__( target_mean, target_variance, initial_mean, context_bounds, perf_lb, init_covar_scale =init_covar_scale,
             max_kl=max_kl, std_lower_bound=std_lower_bound, kl_threshold=kl_threshold)
        # self.ctx_conversion_fn = self._clip_round_ctx


    def sample(self):
        sample = self.context_dist.rsample().detach().numpy()
        return np.clip(np.round(sample), self.context_bounds[0], self.context_bounds[1])
