
from torch.distributions import MultivariateNormal
import numpy as np
import torch



def w_project_norm(x, W, y):
    return np.matmul(np.matmul(x,W),y)
class GaussianSelfPacedTeacher:

    def __init__(self,target_mean, target_variance, initial_mean, initial_var,  context_bounds, perf_lb,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, use_avg_performance=False, perf_slack = 1, lb= 0.01):


        self.max_kl = max_kl
        self.v_bar =0
        self.update_ctr = 0
        self.context_dim = target_mean.shape[0]
        self.context_bounds = context_bounds
        # self.bounds = context_bounds
        self.use_avg_performance = use_avg_performance
        self.perf_lb = perf_lb
        self.perf_lb_reached = False
        self.perf_slack = perf_slack
        self.gamma1=0
        self.gamma2=0
        self.lambda1=0
        self.lambda2=0
        if kl_threshold is None:
            raise RuntimeError("Error! kl threshold need to be set")
        else:

            self.kl_threshold = kl_threshold
        std_upper_bound = (context_bounds[1]- context_bounds[0])/2
        # Create the initial context distribution

        # self.scale = init_covar_scale
        # Create the target distribution

        if target_variance.size==1:
            target_covar = target_variance * np.eye(self.context_dim)
            target_scale = target_variance**self.context_dim
            # self.Sigma_inv = (1/np.clip(target_covar, a_min=std_lower_bound, a_max=std_upper_bound)) * np.eye(self.context_dim)

        elif target_variance.size==self.context_dim:
            target_scale = np.prod(target_variance)
            target_covar = np.diag(target_variance.reshape(-1))
            # self.Sigma_inv = np.linalg.inv(np.diag( np.clip(target_variance, a_min=std_lower_bound, a_max=std_upper_bound)))
        elif target_variance.size==self.context_dim**2:
            target_scale = np.linalg.det(target_variance.reshape(self.context_dim, self.context_dim))
            target_covar = target_variance.reshape(self.context_dim, self.context_dim)
            # self.Sigma_inv = np.linalg.inv(np.diag(np.clip(np.diag(target_covar), a_min=std_lower_bound, a_max=std_upper_bound)))
        else:
                raise RuntimeError("Target variance does not match context space")

        if isinstance(std_lower_bound, (int, float)):
            std_lower_bound= std_lower_bound*np.diag(target_covar)/np.min(np.diag(target_covar))
        elif isinstance(std_lower_bound, np.ndarray):
            if std_lower_bound.size ==1:
                std_lower_bound = std_lower_bound * np.ones_like(np.diag(target_covar))

            elif not std_lower_bound.size ==self.context_dim:
                raise RuntimeError("std lower bound is does not have the right size")
        else:
            raise RuntimeError("std lower bound is not compatible")
            # std_lower_bound = min(np.min(target_variance), 1 / (np.max(target_variance)))
        self.std_lower_bound = std_lower_bound

        self.Sigma = np.diag(np.clip(np.diag(target_covar), a_min = std_lower_bound**2, a_max = std_upper_bound**2))
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        if initial_var.size==self.context_dim**2:

            init_covar = np.diag(initial_var.reshape(self.context_dim,self.context_dim))

        elif (initial_var.size==1 or initial_var.size==self.context_dim):
            init_covar= initial_var

        else:
            raise RuntimeError("Initial variance does not match context space")





        init_scale = np.prod(init_covar)**(1/init_covar.size)
        self.ctx_mean = initial_mean
        self.target_mean = target_mean
        self.target_covar = target_covar

        self.target_cov_inv = self.Sigma_inv
        self.covar_scale_lb = max(min(np.min(np.diag(self.target_covar)/init_covar),np.min(init_covar/np.diag(self.target_covar))  ),0.01)
        self.covar_scale_ub = 1/self.covar_scale_lb
        self.ctx_covar_scale = np.clip(.1*init_scale/(target_scale**(1/self.context_dim)), a_min=self.covar_scale_lb, a_max= self.covar_scale_ub)

        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor(self.ctx_covar_scale * self.target_covar, dtype=torch.float64))
        self.target_dist = MultivariateNormal(loc=torch.as_tensor(self.target_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor( self.target_covar, dtype=torch.float64))

    def target_context_kl(self, numpy=True):
        kl_div = torch.distributions.kl.kl_divergence(self.context_dist,
                                                      self.target_dist).detach()
        if numpy:
            kl_div = np.mean(kl_div.numpy())

        return kl_div

    def save(self, path):
        weights = np.concatenate([self.ctx_mean, self.ctx_covar_scale])
        np.save(path, weights)

    def load(self, path):
        weights = np.load(path)
        self.ctx_mean = weights[0:-1]
        self.ctx_covar_scale = weights[-1]
        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor(self.ctx_covar_scale * self.Sigma, dtype=torch.float64))

    def get_task(self):
        return np.concatenate([self.ctx_mean, self.ctx_covar_scale])

    def set_task(self, task):
        self.ctx_mean = task[0:-1]
        self.ctx_covar_scale = task[-1]
        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor( self.ctx_covar_scale * self.target_covar, dtype=torch.float64))

    def _compute_context_kl(self, ctx_mean, ctx_covar_scale, dist = None):
        if dist==None:
            dist = self.target_dist
        context_dist = MultivariateNormal(loc=torch.as_tensor(ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor(
                                                   ctx_covar_scale * self.target_covar, dtype=torch.float64))
        return torch.distributions.kl.kl_divergence(context_dist,
                                                      dist).detach().numpy()


    def _compute_expected_performance(self, values, contexts, ctx_mean, ctx_covar_scale, ):
        ctx_t =torch.Tensor(contexts)
        context_dist = MultivariateNormal(loc=torch.as_tensor(ctx_mean, dtype=torch.float64),
                                          covariance_matrix=torch.as_tensor(
                                              ctx_covar_scale * self.target_covar, dtype=torch.float64))
        con_ratio_t = torch.exp(context_dist.log_prob(ctx_t) - self.context_dist.log_prob(ctx_t))
        return torch.mean(con_ratio_t * values).detach().numpy()

    def get_context(self):
        return {"mean":self.ctx_mean, "var": np.diag( self.ctx_covar_scale * self.target_covar), "kl_div": self.target_context_kl()}

    def get_status(self):
        return {"mean_diff": np.mean(np.abs(self.ctx_mean - self.target_mean)),
                "var_diff": np.mean(
                    np.abs(np.diag(  self.ctx_covar_scale * self.target_covar- self.target_covar))),
                'perf_lb': self.perf_lb_reached,
                'v_bar': self.v_bar,
                'theta_hat': self.ctx_covar_scale,
                'mean_converged':np.mean((self.ctx_mean - self.target_mean)**2)<.01,
                'covar_converged': np.all(abs(np.diag(self.ctx_covar_scale)-1) < self.std_lower_bound),
                'converged':np.all(abs(np.diag(self.ctx_covar_scale)-1) < self.std_lower_bound) and np.mean((self.ctx_mean - self.target_mean)**2)<.01
                }

    def export_dist(self):

        return dict(target_mean=[self.ctx_mean,],
                    target_var=[ self.covariance_matrix(self.ctx_covar_scale),],
                    target_priors=np.array([1,]))

    def update_mean(self, avg_performance, contexts, values, ):
        V_bar = np.mean(values)
        self.v_bar =V_bar

        u_bar = -np.dot( values,self.ctx_mean - contexts)/len(values)
        u_bar_norm2 = w_project_norm(u_bar, self.target_cov_inv, u_bar)
        self.update_ctr += 1

        if V_bar >= self.perf_lb:
            # self.perf_lb = V_bar * .3 + .7 * self.perf_lb
            # print("Optimizing KL")
            self.perf_lb_reached = True
            # u_bar = np.dot( values, self.ctx_mean - contexts)/len(values)


            du = self.target_mean - self.ctx_mean
            du_norm2 = w_project_norm(du, self.target_cov_inv, du)
            du_u_bar_norm = w_project_norm(du, self.target_cov_inv, u_bar)
            w = self.ctx_covar_scale*(V_bar - self.perf_lb)**2 /(2*self.max_kl)
            cos_phi = du_u_bar_norm / (du_norm2 * u_bar_norm2) ** .5
            gamma1_ = ((w * (1 - cos_phi ** 2) / ( u_bar_norm2-w)) ** 0.5 - cos_phi) * (du_norm2 / u_bar_norm2) ** 0.5 if w< u_bar_norm2 else -1
            delta = du + gamma1_ * u_bar
            gamma2_ = (w_project_norm(delta, self.target_cov_inv, delta) / (2 * self.ctx_covar_scale * self.max_kl)) ** 0.5 if -du_u_bar_norm - gamma1_*u_bar_norm2>0 else -1
            if gamma2_ <1 or gamma1_<0:
                gamma2_ =1
                gamma1_ = -((V_bar - self.perf_lb) + du_u_bar_norm/self.ctx_covar_scale)/u_bar_norm2


            if gamma1_<0:
                gamma1_ =0
                gamma2_ = (du_norm2/ (
                            2 * self.ctx_covar_scale * self.max_kl)) ** 0.5

            if gamma2_<1:

                gamma2_ = 1
            #
            ctx_mean = self.ctx_mean + (du + gamma1_ * u_bar)/gamma2_
            self.gamma2=gamma2_
            self.gamma1=gamma1_
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])

            return


        elif not self.perf_lb_reached:
            ctx_mean = self.ctx_mean + (2*self.max_kl*self.ctx_covar_scale/u_bar_norm2)**.5 *u_bar
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])
            return


    def update_covar_scale(self, avg_performance, contexts, values, ):
        V_bar = np.mean(values)
        self.v_bar =V_bar

        u_bar = np.mean(values*np.sum(np.matmul(self.ctx_mean - contexts, self.target_cov_inv) * (self.ctx_mean - contexts), axis=1))
        self.update_ctr += 1
        if V_bar >= self.perf_lb:
            # print("Optimizing KL")
            self.perf_lb_reached = True
            # self.perf_lb = V_bar*.3 + .7* self.perf_lb
            if self.ctx_covar_scale <1:
                if (u_bar - V_bar*self.context_dim*self.ctx_covar_scale)<0:
                    scale = self.ctx_covar_scale*(1 +self.perf_slack* min(2*(self.max_kl/self.context_dim)**0.5,
                            (2*(V_bar - self.perf_lb)*self.ctx_covar_scale) / (-u_bar + V_bar*self.context_dim*self.ctx_covar_scale)))
                else:
                    scale = self.ctx_covar_scale + 2*self.perf_slack* self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5
                if np.isnan(scale):
                    return
                self.ctx_covar_scale = np.clip(scale, a_min=self.ctx_covar_scale, a_max=1)
                return

            else:

                if (u_bar - V_bar*self.context_dim*self.ctx_covar_scale)>0:
                    scale = self.ctx_covar_scale - self.perf_slack*min(2*self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5,
                            (2*self.ctx_covar_scale**2 *(V_bar - self.perf_lb) )/(u_bar - V_bar*self.context_dim*self.ctx_covar_scale))
                else:
                    scale = self.ctx_covar_scale -2*self.perf_slack*self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5
                if np.isnan(scale):
                    return
                self.ctx_covar_scale = np.clip(scale, a_min=1, a_max=self.ctx_covar_scale)
                return


        elif not self.perf_lb_reached:


            s_ = self.perf_slack*2*self.ctx_covar_scale*(self.max_kl/self.context_dim)**0.5
            s_new = self.ctx_covar_scale+ s_ if (u_bar  - V_bar*self.context_dim*self.ctx_covar_scale)> 0 else self.ctx_covar_scale - s_
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
                                               covariance_matrix=torch.as_tensor(self.ctx_covar_scale * self.target_covar, dtype=torch.float64),
                                               )

    def sample(self):
        sample = self.context_dist.rsample().detach().numpy()
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])




class DiscreteGaussianSelfPacedTeacher(GaussianSelfPacedTeacher):

    def __init__(self, target_mean, initial_mean, context_bounds, perf_lb, init_covar_scale = 0.01,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None):


        target_variance = np.eye(target_mean.shape[0])*0.01
        super(DiscreteGaussianSelfPacedTeacher, self).__init__( target_mean, target_variance, initial_mean, context_bounds, perf_lb, init_covar_scale =init_covar_scale,
             max_kl=max_kl, std_lower_bound=std_lower_bound, kl_threshold=kl_threshold,)
        # self.ctx_conversion_fn = self._clip_round_ctx


    def sample(self):
        sample = self.context_dist.rsample().detach().numpy()
        return np.clip(np.round(sample), self.context_bounds[0], self.context_bounds[1])



class GaussianSelfPacedTeacherV2(GaussianSelfPacedTeacher):

    def covariance_matrix(self, ctx_covar_scale):
        return ctx_covar_scale**.5@self.Sigma@ctx_covar_scale**.5

    def __init__(self, target_mean, target_variance, initial_mean, initial_var,  context_bounds, perf_lb,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, use_avg_performance=False):


        self.max_kl = max_kl
        self.gamma1=0
        self.gamma2=1
        self.u_bar=np.zeros_like(initial_mean)
        self.psi_bar=0
        self.v_bar =0
        self.update_ctr = 0
        self.context_dim = target_mean.shape[0]
        self.context_bounds = context_bounds
        # self.bounds = context_bounds
        self.use_avg_performance = use_avg_performance
        self.perf_lb_reached = False
        self.perf_lb = perf_lb
        if kl_threshold is None:
            raise RuntimeError("Error! kl threshold need to be set")
        else:

            self.kl_threshold = kl_threshold
        std_upper_bound = (context_bounds[1]- context_bounds[0])/2
        # Create the initial context distribution

        # self.scale = init_covar_scale
        # Create the target distribution

        if target_variance.size == 1:
            target_covar = target_variance * np.eye(self.context_dim)
            # target_scale = target_variance**self.context_dim
            # self.Sigma_inv = (1/np.clip(target_covar, a_min=std_lower_bound, a_max=std_upper_bound)) * np.eye(self.context_dim)

        elif target_variance.size==self.context_dim:
            # target_scale = np.prod(target_variance)
            target_covar = np.diag(target_variance.reshape(-1))
            # self.Sigma_inv = np.linalg.inv(np.diag( np.clip(target_variance, a_min=std_lower_bound, a_max=std_upper_bound)))
        elif target_variance.size==self.context_dim**2:
            # target_scale = np.linalg.det(target_variance.reshape(self.context_dim, self.context_dim))
            target_covar = target_variance.reshape(self.context_dim, self.context_dim)
            # self.Sigma_inv = np.linalg.inv(np.diag(np.clip(np.diag(target_covar), a_min=std_lower_bound, a_max=std_upper_bound)))
        else:
                raise RuntimeError("Target variance does not match context space")

        if isinstance(std_lower_bound, (int, float)):
            std_lower_bound= std_lower_bound* np.ones_like(np.diag(target_covar))
        elif isinstance(std_lower_bound, np.ndarray):
            if std_lower_bound.size ==1:
                std_lower_bound = std_lower_bound * np.ones_like(np.diag(target_covar))

            elif not std_lower_bound.size ==self.context_dim:
                raise RuntimeError("std lower bound is does not have the right size")
        else:
            raise RuntimeError("std lower bound is not compatible")
            # std_lower_bound = min(np.min(target_variance), 1 / (np.max(target_variance)))
        self.std_lower_bound = std_lower_bound
        self.Sigma = target_covar.copy()+ np.diag(std_lower_bound ** 2)
        # np.fill_diagonal(self.Sigma , np.clip(np.diag(target_covar), a_min = std_lower_bound**2, a_max = std_upper_bound**2))
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        if initial_var.size==self.context_dim**2:

            init_covar = np.diag(initial_var.reshape(self.context_dim,self.context_dim))

        elif initial_var.size==1:
            init_covar= initial_var*np.ones_like(np.diag(target_covar))
        elif initial_var.size==self.context_dim:
            init_covar= initial_var

        else:
            raise RuntimeError("Initial variance does not match context space")





        self.ctx_covar_scale = np.diag(np.clip(init_covar, a_min = std_lower_bound**2, a_max = std_upper_bound**2)/np.diag(self.Sigma))
        self.ctx_mean = initial_mean
        self.target_mean = target_mean
        self.target_covar = target_covar

        self.target_cov_inv = self.Sigma_inv
        self.covar_scale_lb = np.minimum(np.min(np.diag(self.Sigma), axis=0)/init_covar, init_covar/np.max(np.diag(self.Sigma), axis=0))
        self.covar_scale_ub = 1/ self.covar_scale_lb

        # self.covar_scale_ub = 1/self.covar_scale_lb
        # self.ctx_covar_scale = np.diag(initial_var)
        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor(self.covariance_matrix(self.ctx_covar_scale), dtype=torch.float64))


        self.target_dist = MultivariateNormal(loc=torch.as_tensor(self.target_mean, dtype=torch.float64),
                                                  covariance_matrix=torch.as_tensor(self.target_covar, dtype=torch.float64))





    def load(self, path):
        weights = np.load(path)
        self.ctx_mean = weights[0:self.context_dim]
        self.ctx_covar_scale = np.diag(weights[self.context_dim:])
        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor( self.covariance_matrix(self.ctx_covar_scale), dtype=torch.float64))

    def get_task(self):
        return np.concatenate([self.ctx_mean, np.diag(self.ctx_covar_scale)])

    def set_task(self, task):
        self.ctx_mean = task[0:self.context_dim]
        self.ctx_covar_scale = np.diag(task[self.context_dim:])
        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor( self.covariance_matrix(self.ctx_covar_scale), dtype=torch.float64))

    def _compute_context_kl(self, ctx_mean, ctx_covar_scale, dist = None):
        if dist==None:
            dist = self.target_dist
        context_dist = MultivariateNormal(loc=torch.as_tensor(ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor(self.covariance_matrix(ctx_covar_scale), dtype=torch.float64))
        return torch.distributions.kl.kl_divergence(context_dist,
                                                      dist).detach().numpy()


    def _compute_expected_performance(self, values, contexts, ctx_mean, ctx_covar_scale, ):
        ctx_t =torch.Tensor(contexts)
        context_dist = MultivariateNormal(loc=torch.as_tensor(ctx_mean, dtype=torch.float64),
                                          covariance_matrix=torch.as_tensor(self.covariance_matrix(ctx_covar_scale), dtype=torch.float64))
        con_ratio_t = torch.exp(context_dist.log_prob(ctx_t) - self.context_dist.log_prob(ctx_t))
        return torch.mean(con_ratio_t * values).detach().numpy()

    def get_context(self):
        return {"mean":self.ctx_mean, "var": np.diag( self.covariance_matrix(self.ctx_covar_scale)), "kl_div": self.target_context_kl()}

    def get_status(self):
        return {"mean_diff": np.mean(np.abs(self.ctx_mean - self.target_mean)),
                "var_diff": np.mean(
                    np.abs(np.diag(self.covariance_matrix(self.ctx_covar_scale) - self.target_covar))),
                'perf_lb': self.perf_lb_reached,
                'v_bar': self.v_bar,
                'theta_hat': (np.linalg.det(self.context_dist.covariance_matrix)/ np.linalg.det(self.Sigma))**(1/self.context_dim)}

    def export_dist(self):

        return dict(target_mean=[self.ctx_mean,],
                    target_var=[ self.covariance_matrix(self.ctx_covar_scale),],
                    target_priors=np.array([1,]))


    def update_mean(self, avg_performance, contexts, values, ):
        self.update_ctr += 1

        W = np.diag(1/np.diag(self.ctx_covar_scale)**.5)
        W = W@self.Sigma_inv@W
        # W_inv= self.ctx_covar_scale**.5@self.Sigma@self.ctx_covar_scale**.5
        V_bar = np.mean(values)
        self.v_bar =V_bar

        u_bar = -np.dot( values.T,self.ctx_mean - contexts)/values.size
        u_bar_norm2_W = u_bar@W@ u_bar.T
        self.u_bar= u_bar
        if V_bar >= self.perf_lb:
            # self.perf_lb = V_bar * .3 + .7 * self.perf_lb
            # print("Optimizing KL")
            self.perf_lb_reached = True
            # u_bar = np.dot( values, self.ctx_mean - contexts)/len(values)


            du = self.target_mean - self.ctx_mean
            du_norm2 =du@W@du.T
            du_u_bar_norm = du@W@ u_bar
            # w = self.ctx_covar_scale*(V_bar - self.perf_lb)**2 /(2*self.max_kl)
            # cos_phi = du_u_bar_norm / (du_norm2 * u_bar_norm2) ** .5
            # gamma1_ = ((w * (1 - cos_phi ** 2) / ( u_bar_norm2-w)) ** 0.5 - cos_phi) * (du_norm2 / u_bar_norm2) ** 0.5 if w< u_bar_norm2 else -1
            # delta = du + gamma1_ * u_bar
            # gamma2_ = (w_project_norm(delta, self.target_cov_inv, delta) / (2 * self.ctx_covar_scale * self.max_kl)) ** 0.5 if -du_u_bar_norm - gamma1_*u_bar_norm2>0 else -1
            delta = (du_u_bar_norm**2 - u_bar_norm2_W*(du_norm2 - 2*self.max_kl))**.5 if (du_u_bar_norm**2 - u_bar_norm2_W*(du_norm2 - 2*self.max_kl))>0 else - u_bar_norm2_W
            gamma2_ = (du_norm2/(2*self.max_kl))**.5
            gamma1_ = - ((V_bar - self.perf_lb) + du_u_bar_norm)/(u_bar_norm2_W)
            if (V_bar - self.perf_lb) + du_u_bar_norm>=0 and du_norm2<= 2*self.max_kl:
                gamma1_ = 0
                gamma2_ = 1
            elif gamma2_>= - (du_u_bar_norm)/(V_bar - self.perf_lb) and gamma2_>=1:
                gamma1_ = 0
            elif (- u_bar_norm2_W +delta)/(u_bar_norm2_W) >= gamma1_ >= (- u_bar_norm2_W - delta)/(u_bar_norm2_W) and gamma1_>=0:
                gamma2_ = 1
            else:
                gamma2_ = ((du_norm2*u_bar_norm2_W - du_u_bar_norm)/(2*self.max_kl* u_bar_norm2_W - (V_bar - self.perf_lb)**2))**.5 if (2*self.max_kl* u_bar_norm2_W - (V_bar - self.perf_lb)**2)>0 else 100*du_norm2
                gamma1_ = -((V_bar - self.perf_lb)* gamma2_ + du_u_bar_norm)/(u_bar_norm2_W)


            ctx_mean = self.ctx_mean + (du + gamma1_ * u_bar)/gamma2_

            self.gamma1 = gamma1_
            self.gamma2 = gamma2_
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])



        elif not self.perf_lb_reached:
            ctx_mean = self.ctx_mean + (2*self.max_kl/u_bar_norm2_W)**.5 *u_bar
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])

        return


    def update_covar_scale(self, avg_performance, contexts, values, ):
        theta_ = 1/np.diag(self.ctx_covar_scale)

        omega =np.diag(theta_**1.5)@ self.Sigma_inv @ np.diag(theta_**.5)
        # w = 1- 1/self.ctx_covar_scale
        # W = theta@self.Sigma_inv@theta

        V_bar = np.mean(values)
        self.v_bar =V_bar

        # omega = np.matmul(self.ctx_mean - contexts, self.target_cov_inv) * (self.ctx_mean - contexts)
        u_bar = ((contexts - self.ctx_mean)@omega)*(contexts - self.ctx_mean)
        u_bar = -.5* np.dot(u_bar.T, values)/ values.size + 0.5*V_bar*theta_
        # u_bar = - np.eye(self.context_dim) + u_bar*np.eye(self.context_dim)
        # u_bar = (.5/values.size)*(np.dot(values, 1/ self.ctx_covar_scale + omega/self.ctx_covar_scale**2 ))
        H = (np.diag(theta_)@(self.Sigma_inv*self.Sigma)@np.diag(theta_) + np.diag(theta_**2))/2
        H_inv = np.linalg.inv(H)
        u_bar_Hi_norm2 = u_bar @ H_inv @ u_bar.T

        self.update_ctr += 1
        if V_bar >= self.perf_lb:
            # print("Optimizing KL")
            self.perf_lb_reached = True
            # self.perf_lb = V_bar*.3 + .7* self.perf_lb
            delta_V = V_bar - self.perf_lb
            w= np.diag(theta_**1.5)@ self.Sigma@ self.Sigma_inv @ np.diag(theta_**.5)
            theta = np.diag(self.ctx_covar_scale)
            w = 0.5*(theta_ -  np.diag(w))
            d_theta= np.ones_like(theta) - theta

            if delta_V+ np.dot(d_theta, u_bar)>= 0 and (d_theta@H@ d_theta.T) <= 4*self.max_kl: # unconstrained solution
                self.ctx_covar_scale = np.eye(self.context_dim)
                return
            else:
                w_Hi_norm2 = w @ H_inv @ w.T
                u_bar_Hi_w = u_bar@H_inv@w.T
                gamma2 = .5*(w_Hi_norm2/self.max_kl)**.5
                if gamma2>u_bar_Hi_w/delta_V:
                    gamma1 = 0
                else:
                    gamma2 = ((w_Hi_norm2* u_bar_Hi_norm2 - u_bar_Hi_w**2)/(4*self.max_kl - delta_V**2))**.5 if (w_Hi_norm2* u_bar_Hi_norm2 - u_bar_Hi_w**2)/(4*self.max_kl - delta_V**2)>0 else 100*u_bar_Hi_norm2
                    gamma1 = (u_bar_Hi_w - gamma2*delta_V)/(u_bar_Hi_norm2)


            scale = theta + (1/gamma2)* (H_inv@(gamma1 * u_bar - w))

            if np.isnan(np.sum(scale)):
                return
            self.ctx_covar_scale = np.diag(np.clip(scale, a_min = self.covar_scale_lb, a_max = self.covar_scale_ub))
            return



        elif not self.perf_lb_reached:


            s_new = np.diag(self.ctx_covar_scale) + 2*self.max_kl**.5/(u_bar_Hi_norm2) * (H_inv@u_bar)
            if np.isnan(np.sum(s_new)):
                return
            self.ctx_covar_scale = np.diag(np.clip(s_new, a_min = self.covar_scale_lb, a_max = self.covar_scale_ub))
        return

    def update_distribution(self, avg_performance, contexts, values, ):
        if self.update_ctr%2 ==0:
            self.update_mean(avg_performance=avg_performance, contexts= contexts, values=values)
        else:
            self.update_covar_scale(avg_performance=avg_performance, contexts= contexts, values=values)

        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor(self.covariance_matrix(self.ctx_covar_scale), dtype=torch.float64),
                                               )

class GaussianSelfPacedTeacherV3(GaussianSelfPacedTeacherV2):
    def covariance_matrix(self, ctx_covar_scale):
        return (ctx_covar_scale**.5)@self.target_covar@(ctx_covar_scale**.5)


    def update_perf_lb(self, iter=0, **kwargs ):
        self.perf_lb= self.perf_lb_fn(iter, self.ctx_mean)
        return
    def update_distribution(self, avg_performance, contexts, values, ):
        self.update_perf_lb()
        if np.random.rand()>0.5:
            self.update_mean(avg_performance=avg_performance, contexts= contexts, values=values)
        else:
            self.update_covar_scale(avg_performance=avg_performance, contexts= contexts, values=values)





        self.performance=self._compute_expected_performance(values=values, contexts=contexts,ctx_mean=self.ctx_mean,ctx_covar_scale=self.ctx_covar_scale)
        self.proximity=self._compute_context_kl(self.ctx_mean,self.ctx_covar_scale,dist=self.context_dist)

        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                           covariance_matrix=torch.as_tensor(
                                               self.covariance_matrix(self.ctx_covar_scale), dtype=torch.float64),
                                           )
    def __init__(self, target_mean, target_variance, initial_mean, initial_var, context_bounds, perf_lb,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, use_avg_performance=False, perf_lb_fn=None):

        self.proximity=0
        self.performance=0
        self.max_kl = max_kl
        self.v_bar = 0
        self.update_ctr = 0
        self.context_dim = target_mean.shape[0]
        self.context_bounds = context_bounds
        # self.bounds = context_bounds
        self.use_avg_performance = use_avg_performance
        self.perf_lb = perf_lb
        self.perf_lb_reached = False
        self.u_bar= np.zeros_like(initial_mean)
        self.gamma1=0
        self.gamma2=1
        self.lambda2=0
        self.lambda1=0
        if kl_threshold is None:
            raise RuntimeError("Error! kl threshold need to be set")
        else:

            self.kl_threshold = kl_threshold
        std_upper_bound = (context_bounds[1] - context_bounds[0]) / 2
        # Create the initial context distribution

        # self.scale = init_covar_scale
        # Create the target distribution

        if target_variance.size == 1:
            target_covar = target_variance * np.eye(self.context_dim)
            # target_scale = target_variance**self.context_dim
            # self.Sigma_inv = (1/np.clip(target_covar, a_min=std_lower_bound, a_max=std_upper_bound)) * np.eye(self.context_dim)

        elif target_variance.size == self.context_dim:
            # target_scale = np.prod(target_variance)
            target_covar = np.diag(target_variance.reshape(-1))
            # self.Sigma_inv = np.linalg.inv(np.diag( np.clip(target_variance, a_min=std_lower_bound, a_max=std_upper_bound)))
        elif target_variance.size == self.context_dim ** 2:
            # target_scale = np.linalg.det(target_variance.reshape(self.context_dim, self.context_dim))
            target_covar = target_variance.reshape(self.context_dim, self.context_dim)
            # self.Sigma_inv = np.linalg.inv(np.diag(np.clip(np.diag(target_covar), a_min=std_lower_bound, a_max=std_upper_bound)))
        else:
            raise RuntimeError("Target variance does not match context space")

        if isinstance(std_lower_bound, (int, float)):
            std_lower_bound = std_lower_bound * np.ones_like(np.diag(target_covar))
        elif isinstance(std_lower_bound, np.ndarray):
            if std_lower_bound.size == 1:
                std_lower_bound = std_lower_bound * np.ones_like(np.diag(target_covar))

            elif not std_lower_bound.size == self.context_dim:
                raise RuntimeError("std lower bound is does not have the right size")
        else:
            raise RuntimeError("std lower bound is not compatible")
            # std_lower_bound = min(np.min(target_variance), 1 / (np.max(target_variance)))
        self.std_lower_bound = std_lower_bound
        self.Sigma = target_covar.copy()
        self.E = np.diag(std_lower_bound ** 2)
        # np.fill_diagonal(self.Sigma,
        #                  np.clip(np.diag(target_covar), a_min=std_lower_bound ** 2, a_max=std_upper_bound ** 2))
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        if initial_var.size == self.context_dim ** 2:

            init_covar = np.diag(initial_var.reshape(self.context_dim, self.context_dim))

        elif initial_var.size == 1:
            init_covar = initial_var * np.ones_like(np.diag(target_covar))
        elif initial_var.size == self.context_dim:
            init_covar = initial_var

        else:
            raise RuntimeError("Initial variance does not match context space")


        self.ctx_mean = initial_mean
        self.target_mean = target_mean
        self.target_covar = target_covar

        self.target_cov_inv = np.linalg.inv(self.target_covar)
        self.covar_scale_lb = np.minimum(np.min(np.diag(self.target_covar), axis=0) / init_covar,
                                         init_covar / np.max(np.diag(self.target_covar), axis=0))
        self.covar_scale_ub = 1 / self.covar_scale_lb
        self.theta_lb = np.minimum(np.min(np.diag(self.Sigma), axis=0) / init_covar,
                                         init_covar / np.max(np.diag(self.Sigma), axis=0))
        self.theta_ub = 1 / self.theta_lb
        # self.covar_scale_ub = 1/self.covar_scale_lb
        # self.ctx_covar_scale = np.diag(initial_var)
        self.ctx_covar_scale = np.diag(
            np.clip(init_covar / np.diag(target_covar),
                    a_min = self.covar_scale_lb, a_max = self.covar_scale_ub))
        self.context_dist = MultivariateNormal(loc=torch.as_tensor(self.ctx_mean, dtype=torch.float64),
                                               covariance_matrix=torch.as_tensor(
                                                   self.covariance_matrix(self.ctx_covar_scale), dtype=torch.float64))

        self.target_dist = MultivariateNormal(loc=torch.as_tensor(self.target_mean, dtype=torch.float64),
                                              covariance_matrix=torch.as_tensor(self.target_covar, dtype=torch.float64))

        self.psi_bar= np.zeros_like(self.theta_lb)

        self.perf_lb_fn = lambda itr, ctx_mean: perf_lb if perf_lb_fn is None else perf_lb_fn(itr, ctx_mean)



    def update_mean(self, avg_performance, contexts, values, ):
        self.update_ctr += 1
    ## v11.0.2
        # W = np.diag(1/np.clip(np.diag(self.ctx_covar_scale), a_min=self.theta_lb, a_max=self.theta_ub)**.5)
        # W = W@self.Sigma_inv@W

        ##V11.1.0
        W = self.covariance_matrix(self.ctx_covar_scale) + self.E
        W = np.linalg.inv(W)

        # W_inv= self.ctx_covar_scale**.5@self.Sigma@self.ctx_covar_scale**.5
        V_bar = np.mean(values)
        self.v_bar =V_bar

        u_bar = -np.dot( values.T,self.ctx_mean - contexts)/values.size
        u_bar_norm2_W = u_bar@W@ u_bar.T

        if V_bar >= self.perf_lb:
            # self.perf_lb = V_bar * .3 + .7 * self.perf_lb
            # print("Optimizing KL")
            self.perf_lb_reached = True
            # u_bar = np.dot( values, self.ctx_mean - contexts)/len(values)


            du = self.target_mean - self.ctx_mean
            du_norm2 =du@W@du.T
            du_u_bar_norm = du@W@ u_bar
            # w = self.ctx_covar_scale*(V_bar - self.perf_lb)**2 /(2*self.max_kl)
            # cos_phi = du_u_bar_norm / (du_norm2 * u_bar_norm2) ** .5
            # gamma1_ = ((w * (1 - cos_phi ** 2) / ( u_bar_norm2-w)) ** 0.5 - cos_phi) * (du_norm2 / u_bar_norm2) ** 0.5 if w< u_bar_norm2 else -1
            # delta = du + gamma1_ * u_bar
            # gamma2_ = (w_project_norm(delta, self.target_cov_inv, delta) / (2 * self.ctx_covar_scale * self.max_kl)) ** 0.5 if -du_u_bar_norm - gamma1_*u_bar_norm2>0 else -1
            delta = ( du_u_bar_norm**2 - u_bar_norm2_W*(du_norm2 - 2*self.max_kl))**.5 if (du_u_bar_norm**2 - u_bar_norm2_W*(du_norm2 - 2*self.max_kl))>0 else - u_bar_norm2_W
            gamma2_ = (du_norm2/(2*self.max_kl))**.5
            gamma1_ = - ((V_bar - self.perf_lb) + du_u_bar_norm)/(u_bar_norm2_W)
            if (V_bar - self.perf_lb) + du_u_bar_norm>=0 and du_norm2<= 2*self.max_kl:
                gamma1_ = 0
                gamma2_ = 1
            elif gamma1_<= (- u_bar_norm2_W +delta)/(u_bar_norm2_W) and gamma1_>= (- u_bar_norm2_W -delta)/(u_bar_norm2_W) and gamma1_>=0:
                gamma2_ = 1
            elif gamma2_>= - (du_u_bar_norm)/(V_bar - self.perf_lb) and gamma2_>=1:
                gamma1_ = 0

            else:
                gamma2_ = ((du_norm2*u_bar_norm2_W - du_u_bar_norm)/(2*self.max_kl* u_bar_norm2_W - (V_bar - self.perf_lb)**2))**.5 if (2*self.max_kl* u_bar_norm2_W - (V_bar - self.perf_lb)**2)>0 else 100*du_norm2
                gamma1_ = -((V_bar - self.perf_lb)* gamma2_ + du_u_bar_norm)/(u_bar_norm2_W)


            ctx_mean = self.ctx_mean + (du + gamma1_ * u_bar)/gamma2_


            self.u_bar = u_bar
            self.gamma1= gamma1_
            self.gamma2= gamma2_
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])



        elif not self.perf_lb_reached:
            ctx_mean = self.ctx_mean + (2*self.max_kl/u_bar_norm2_W)**.5 *u_bar
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])

        return


    def update_covar_scale(self, avg_performance, contexts, values, ):
        theta_ = 1/np.diag(self.ctx_covar_scale)

        omega =np.diag(theta_**1.5)@ self.target_cov_inv @ np.diag(theta_**.5)
        # w = 1- 1/self.ctx_covar_scale
        # W = theta@self.Sigma_inv@theta

        V_bar = np.mean(values)
        self.v_bar =V_bar

        # omega = np.matmul(self.ctx_mean - contexts, self.target_cov_inv) * (self.ctx_mean - contexts)
        u_bar = ((contexts - self.ctx_mean)@omega)*(contexts - self.ctx_mean)
        u_bar = .5* np.dot(u_bar.T, values)/ values.size -0.5*V_bar*theta_
        # u_bar = - np.eye(self.context_dim) + u_bar*np.eye(self.context_dim)
        # u_bar = (.5/values.size)*(np.dot(values, 1/ self.ctx_covar_scale + omega/self.ctx_covar_scale**2 ))
        H = (np.diag(theta_)@(self.target_cov_inv*self.target_covar)@np.diag(theta_) + np.diag(theta_**2))/2
        H_inv = np.linalg.inv(H)
        u_bar_Hi_norm2 = u_bar @ H_inv @ u_bar.T

        self.update_ctr += 1
        if V_bar >= self.perf_lb:
            # print("Optimizing KL")
            self.perf_lb_reached = True
            # self.perf_lb = V_bar*.3 + .7* self.perf_lb
            delta_V = V_bar - self.perf_lb
            theta = np.diag(self.ctx_covar_scale)
            w= - np.diag(theta_**1.5)@ (self.target_covar* self.target_cov_inv) @ (theta_**.5) #- (np.diag(theta_**1.5)@self.target_cov_inv@np.diag(theta_**.5)@(self.target_mean - self.ctx_mean).T)* (self.target_mean - self.ctx_mean)

            w = 0.5*(theta_ +  w)
            d_theta= np.ones_like(theta) - theta

            if delta_V+ np.dot(d_theta, u_bar)>= 0 and (d_theta@H@ d_theta.T) <= 4*self.max_kl: # unconstrained solution
                self.ctx_covar_scale = np.eye(self.context_dim)
                return
            else:
                w_Hi_norm2 = w @ H_inv @ w.T
                u_bar_Hi_w = u_bar@H_inv@w.T
                gamma2 = .5*(w_Hi_norm2/self.max_kl)**.5
                if gamma2>u_bar_Hi_w/delta_V:
                    gamma1 = 0
                else:
                    gamma2 = ((w_Hi_norm2* u_bar_Hi_norm2 - u_bar_Hi_w**2)/(4*self.max_kl - delta_V**2))**.5 if (w_Hi_norm2* u_bar_Hi_norm2 - u_bar_Hi_w**2)/(4*self.max_kl - delta_V**2)>0 else 100*u_bar_Hi_norm2
                    gamma1 = (u_bar_Hi_w - gamma2*delta_V)/(u_bar_Hi_norm2)
                    if gamma2 <0 or gamma1<0:
                        return

            self.lambda1=gamma1
            self.lambda2=gamma2
            self.psi_bar= u_bar
            scale = theta + (1/gamma2)* (H_inv@(gamma1 * u_bar - w))

            if np.isnan(np.sum(scale)):
                return
            self.ctx_covar_scale = np.diag(np.clip(scale, a_min = self.covar_scale_lb, a_max = self.covar_scale_ub))
            return



        elif not self.perf_lb_reached:


            s_new = np.diag(self.ctx_covar_scale) + 2*(self.max_kl**.5/u_bar_Hi_norm2)**0.5* (H_inv@u_bar)
            if np.isnan(np.sum(s_new)):
                return
            self.ctx_covar_scale = np.diag(np.clip(s_new, a_min = self.covar_scale_lb, a_max = self.covar_scale_ub))
        return

    def get_status(self):
        delta_mu = self.ctx_mean - self.target_mean
        info={"mean_diff": np.mean(np.abs(delta_mu)),
         "var_diff": np.mean(
             np.abs(np.diag(self.covariance_matrix(self.ctx_covar_scale) - self.target_covar))),
         'perf_lb_reached': self.perf_lb_reached,
         'v_bar': self.v_bar,
         'perf_lb': self.perf_lb,
         'delta_V': self.v_bar - self.perf_lb,
         'expected_performance': self.performance,
         'context_kl': self.proximity,
         'theta_hat': np.linalg.det(self.ctx_covar_scale),
         'mean_convergence': delta_mu @ self.Sigma_inv @ delta_mu.T,
         'covar_convergence': np.mean((np.diag(self.ctx_covar_scale) - 1) ** 2),
         'converged': np.all(np.diag(
             self.covariance_matrix(self.ctx_covar_scale) - self.target_covar) < self.std_lower_bound) and np.mean(
             (self.ctx_mean - self.target_mean) ** 2) < .01,
         'gamma1_': self.gamma1,
         'gamma2_': self.gamma2,
         'lambda1_': self.lambda1,
         'lambda2_': self.lambda2,

         }
        u_bar = {f'u_bar_{i}': self.u_bar[i] for i in range(delta_mu.size)}
        psi_bar = {f'psi_bar_{i}': self.psi_bar[i] for i in range(delta_mu.size)}
        info.update(u_bar)
        info.update(psi_bar)

        return info

class GaussianSelfPacedTeacherV4(GaussianSelfPacedTeacherV3):
    def __init__(self, target_mean, target_variance, initial_mean, initial_var, context_bounds, perf_lb,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, use_avg_performance=False, perf_lb_fn=None, ctx_mean_transform= None):
        super().__init__( target_mean=target_mean, target_variance=target_variance, initial_mean=initial_mean, initial_var=initial_var,
                         context_bounds=context_bounds, perf_lb=perf_lb,
                         max_kl=max_kl, std_lower_bound=std_lower_bound, kl_threshold=kl_threshold, use_avg_performance=use_avg_performance, perf_lb_fn=perf_lb_fn)

        if ctx_mean_transform is None:
            self.ctx_transform = lambda x: x
        else:
            self.ctx_transform = ctx_mean_transform
    def update_mean(self, avg_performance, contexts, values, ):
        self.update_ctr += 1
    ## v11.0.2
        # W = np.diag(1/np.clip(np.diag(self.ctx_covar_scale), a_min=self.theta_lb, a_max=self.theta_ub)**.5)
        # W = W@self.Sigma_inv@W

        ##V11.1.0
        W = self.covariance_matrix(self.ctx_covar_scale) + self.E
        W = np.linalg.inv(W)

        # W_inv= self.ctx_covar_scale**.5@self.Sigma@self.ctx_covar_scale**.5
        V_bar = np.mean(values)
        self.v_bar =V_bar

        u_bar = -np.dot( values.T,self.ctx_mean - contexts)/values.size
        u_bar_norm2_W = u_bar@W@ u_bar.T
        self.u_bar = u_bar

        if V_bar >= self.perf_lb:
            # self.perf_lb = V_bar * .3 + .7 * self.perf_lb
            # print("Optimizing KL")
            self.perf_lb_reached = True
            # u_bar = np.dot( values, self.ctx_mean - contexts)/len(values)


            du = self.target_mean - self.ctx_mean
            du_norm2 =du@W@du.T
            du_u_bar_norm = du@W@ u_bar
            # w = self.ctx_covar_scale*(V_bar - self.perf_lb)**2 /(2*self.max_kl)
            # cos_phi = du_u_bar_norm / (du_norm2 * u_bar_norm2) ** .5
            # gamma1_ = ((w * (1 - cos_phi ** 2) / ( u_bar_norm2-w)) ** 0.5 - cos_phi) * (du_norm2 / u_bar_norm2) ** 0.5 if w< u_bar_norm2 else -1
            # delta = du + gamma1_ * u_bar
            # gamma2_ = (w_project_norm(delta, self.target_cov_inv, delta) / (2 * self.ctx_covar_scale * self.max_kl)) ** 0.5 if -du_u_bar_norm - gamma1_*u_bar_norm2>0 else -1
            delta = ( du_u_bar_norm**2 - u_bar_norm2_W*(du_norm2 - 2*self.max_kl))**.5 if (du_u_bar_norm**2 - u_bar_norm2_W*(du_norm2 - 2*self.max_kl))>0 else - u_bar_norm2_W
            gamma2_ = (du_norm2/(2*self.max_kl))**.5
            gamma1_ = 0
            if  du_norm2<= 2*self.max_kl:
                gamma2_ = 1




            ctx_mean = self.ctx_mean + (du + gamma1_ * u_bar)/gamma2_


            self.gamma1= gamma1_
            self.gamma2= gamma2_
            if np.isnan(np.sum(ctx_mean)):
                return
            self.ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])
            self.ctx_mean=self.ctx_transform(ctx_mean)


        elif not self.perf_lb_reached:
            ctx_mean = self.ctx_mean + (2*self.max_kl/u_bar_norm2_W)**.5 *u_bar
            if np.isnan(np.sum(ctx_mean)):
                return
            ctx_mean = np.clip(ctx_mean, a_min=self.context_bounds[0], a_max=self.context_bounds[1])
            self.ctx_mean=self.ctx_transform(ctx_mean)
        return


    def update_covar_scale(self, avg_performance, contexts, values, ):
        theta_ = 1/np.diag(self.ctx_covar_scale)

        omega =np.diag(theta_**1.5)@ self.target_cov_inv @ np.diag(theta_**.5)
        # w = 1- 1/self.ctx_covar_scale
        # W = theta@self.Sigma_inv@theta

        V_bar = np.mean(values)
        self.v_bar =V_bar

        # omega = np.matmul(self.ctx_mean - contexts, self.target_cov_inv) * (self.ctx_mean - contexts)
        u_bar = ((contexts - self.ctx_mean)@omega)*(contexts - self.ctx_mean)
        u_bar = .5* np.dot(u_bar.T, values)/ values.size -0.5*V_bar*theta_
        # u_bar = - np.eye(self.context_dim) + u_bar*np.eye(self.context_dim)
        # u_bar = (.5/values.size)*(np.dot(values, 1/ self.ctx_covar_scale + omega/self.ctx_covar_scale**2 ))
        H = (np.diag(theta_)@(self.target_cov_inv*self.target_covar)@np.diag(theta_) + np.diag(theta_**2))/2
        H_inv = np.linalg.inv(H)
        u_bar_Hi_norm2 = u_bar @ H_inv @ u_bar.T
        self.psi_bar = u_bar

        self.update_ctr += 1
        if V_bar >= self.perf_lb:
            # print("Optimizing KL")
            self.perf_lb_reached = True
            # self.perf_lb = V_bar*.3 + .7* self.perf_lb
            delta_V = V_bar - self.perf_lb
            theta = np.diag(self.ctx_covar_scale)
            w= - np.diag(theta_**1.5)@ (self.target_covar* self.target_cov_inv) @ (theta_**.5) #- (np.diag(theta_**1.5)@self.target_cov_inv@np.diag(theta_**.5)@(self.target_mean - self.ctx_mean).T)* (self.target_mean - self.ctx_mean)

            w = 0.5*(theta_ +  w)
            d_theta= np.ones_like(theta) - theta

            if  (d_theta@H@ d_theta.T) <= 4*self.max_kl: # unconstrained solution
                self.lambda1 = 0
                self.lambda2 = 0
                self.ctx_covar_scale = np.eye(self.context_dim)
                return
            else:
                w_Hi_norm2 = w @ H_inv @ w.T
                u_bar_Hi_w = u_bar@H_inv@w.T
                gamma2 = .5*(w_Hi_norm2/self.max_kl)**.5
                # if gamma2>u_bar_Hi_w/delta_V:
                #     gamma1 = 0
                # else:
                #     gamma2 = ((w_Hi_norm2* u_bar_Hi_norm2 - u_bar_Hi_w**2)/(4*self.max_kl - delta_V**2))**.5 if (w_Hi_norm2* u_bar_Hi_norm2 - u_bar_Hi_w**2)/(4*self.max_kl - delta_V**2)>0 else 100*u_bar_Hi_norm2
                #     gamma1 = (u_bar_Hi_w - gamma2*delta_V)/(u_bar_Hi_norm2)
                #     if gamma2 <0 or gamma1<0:
                #         return

            self.lambda1=0
            self.lambda2=gamma2
            scale = theta + (1/gamma2)* (H_inv@(0 * u_bar - w))

            if np.isnan(np.sum(scale)):
                return
            self.ctx_covar_scale = np.diag(np.clip(scale, a_min = self.covar_scale_lb, a_max = self.covar_scale_ub))
            return



        elif not self.perf_lb_reached:


            s_new = np.diag(self.ctx_covar_scale) + 2*(self.max_kl**.5/u_bar_Hi_norm2)**0.5* (H_inv@u_bar)
            if np.isnan(np.sum(s_new)):
                return
            self.ctx_covar_scale = np.diag(np.clip(s_new, a_min = self.covar_scale_lb, a_max = self.covar_scale_ub))
        return
