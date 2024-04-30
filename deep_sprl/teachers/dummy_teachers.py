import numpy as np


class GaussianSampler:

    def __init__(self, mean, variance, bounds):
        self.bounds = bounds
        self._mean = mean
        self._covariance = variance if isinstance(variance, np.ndarray) else np.eye(self._mean.shape[0]) * variance
        # print(self.covariance.shape)
    def sample(self):

        sample = np.random.multivariate_normal(self._mean, self._covariance)
        # print(sample.shape,self._mean,self.bounds)

        return np.clip(sample, self.bounds[0], self.bounds[1])

    def mean(self):
        return self._mean.copy()

    def covariance_matrix(self):
        return self._covariance.copy()

class GMMSampler:
    def __init__(self, means, sigma2, w, bounds):
        self.w = w/np.sum(w)
        self.w0 = np.cumsum(w)/w.sum()

        if len(sigma2.shape)<2:
            sigma2 = np.expand_dims(sigma2,axis=0)
            means = np.expand_dims(means,axis=0)
        self.vars= [np.diag(var) for var in sigma2]
        # print(self.vars)
        self.mu = means
        self.bounds = bounds
        self._mean = 0
        s2 = 0
        m2 =0
        for i in range(len(self.w)):
            self._mean += self.w[i]*self.mu[i]
            s2 +=self.w[i]*self.vars[i]
            m2 += self.w[i]*np.diag((self.mu[i])**2)

        self._cov = s2 +m2 - np.diag(self._mean**2 )

    def sample(self):
        i = np.where(np.cumsum(self.w0) > np.random.rand())[0][0]

        return np.clip(np.random.default_rng().multivariate_normal(self.mu[i],self.vars[i]) , self.bounds[0], self.bounds[1])
    def update_distribution(self, mean_disc_rew, contexts,
                        rewards):
        return

    def mean(self):
        self._mean = 0
        s2 = 0
        m2 =0
        for i in range(len(self.w)):
            self._mean += self.w[i]*self.mu[i]
            s2 +=self.w[i]*self.vars[i]
            m2 += self.w[i]*np.diag((self.mu[i])**2)

        self._cov = s2 +m2 - np.diag(self._mean**2 )
        return self._mean

    def covariance_matrix(self):
        self._mean = 0
        s2 = 0
        m2 =0
        for i in range(len(self.w)):
            self._mean += self.w[i]*self.mu[i]
            s2 +=self.w[i]*self.vars[i]
            m2 += self.w[i]*np.diag((self.mu[i])**2)

        self._cov = s2 +m2 - np.diag(self._mean**2 )
        return self._cov

    def report_task(self):
        self._mean = 0
        s2 = 0
        m2 =0
        for i in range(len(self.w)):
            self._mean += self.w[i]*self.mu[i]
            s2 +=self.w[i]*self.vars[i]
            m2 += self.w[i]*np.diag((self.mu[i])**2)

        self._cov = s2 +m2 - np.diag(self._mean**2 )
        return self._cov


    def set_means(self, means):
        self.mu = means
        return

    def set_w(self, w):
        self.w = w / np.sum(w)

        self.w0 = np.cumsum(w) / w.sum()
        return

    def set_vars(self,sigma2):
        self.vars = sigma2
        return

    def reconfig(self, config):
        means = config.get('target_mean', np.clip(np.zeros_like(self.bounds[0]),self.bounds[0],self.bounds[1]))
        sigma2 = config.get('target_var', np.ones_like(self.bounds[0]) / 1000)

        w = config.get('target_priors', np.ones(1 if len(means.shape)==1 else means.shape[0]))
        self.w = w / np.sum(w)
        self.w0 = np.cumsum(w) / w.sum()

        if len(sigma2.shape) < 2:
            sigma2 = np.expand_dims(sigma2, axis=0)
            means = np.expand_dims(means, axis=0)
        self.vars = [np.diag(var) for var in sigma2]
        # print(self.vars)
        self.mu = means
        self._mean = 0
        s2 = 0
        m2 = 0
        for i in range(len(self.w)):
            self._mean += self.w[i] * self.mu[i]
            s2 += self.w[i] * self.vars[i]
            m2 += self.w[i] * np.diag((self.mu[i]) ** 2)

        self._cov = s2 + m2 - np.diag(self._mean ** 2)
        return

    def export_config(self):

        return dict(target_mean = self.mu , target_var = self.vars, target_priors = self.w)


class UniformSampler:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self):
        norm_sample = np.random.uniform(low=-1, high=1, size=self.lower_bound.shape)
        return self._scale_context(norm_sample)

    def mean(self):
        return 0.5 * self.lower_bound + 0.5 * self.upper_bound

    def covariance_matrix(self):
        return np.diag((0.5 * (self.upper_bound - self.lower_bound)) ** 2)

    def _scale_context(self, context):
        b = 0.5 * (self.upper_bound + self.lower_bound)
        m = 0.5 * (self.upper_bound - self.lower_bound)
        return m * context + b

