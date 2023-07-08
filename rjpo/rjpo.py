import numpy as np
from fastprogress import progress_bar

from .cg import relative_resigual_cg



def sample_eta(mu, Ls):
    """Draws a sample from $\mathcal{N}(Q \mu, Q)$. Here $Q$ is assumed to be of the form $Q = \sum_{i=1}^p L_i^T L_i$,
    where the $L_i$ are collected in Ls."""
    
    # Setup
    p = len(Ls)
    n = Ls[0].shape[1]

    # Draw sample with correct covariance
    sample = np.zeros(n)
    for L in Ls:
        sample += L.rmatvec( np.random.normal(size=L.shape[0]) )

    # Add mean
    for L in Ls:
        sample += L.rmatvec( L.matvec( mu ) )

    return sample




class RJPOSampler:

    def __init__(self, Ls, mu=None):
        """Here mu is the mean vector of the Gaussian, and $Ls$ is a list of the $L_i$ such thath
        $Q = \sum_{i=1}^p L_i^T L_i $.
        """

        self.Ls = Ls
        self.n = self.Ls[0].shape[1]
        self.Q = Ls[0].T @ Ls[0]
        if len(self.Ls) > 1:
            for j in range(1, len(self.Ls)):
                self.Q += self.Ls[j].T @ self.Ls[j]

        if mu is None:
            self.mu = np.zeros(self.n)
        else:
            self.mu = mu

    
    def sample(self, n_samples=100, cg_eps=1e-2, x0=None, keep_trace=False, cg_maxits=500, ar_step=True):
        """Runs the RJPO sampler."""

        # Initialize
        if x0 is None:
            x = np.zeros(self.n)
        else:
            x = x0

        if keep_trace: 
            trace = np.zeros((n_samples, self.n))
            trace[0,:] = x

        # Tracking
        n_accepted = 0
        tot_cg_its = 0

        for j in progress_bar(range(1, n_samples)):
            
            # Sample eta
            eta = sample_eta(self.mu, self.Ls)

            # Solve approximately
            x_approx_solve = relative_resigual_cg(self.Q, eta, x0=-x, maxits=cg_maxits, eps=cg_eps)
            x_approx = x_approx_solve["x"]
            tot_cg_its += x_approx_solve["iterations"]

            # Compute residual
            residual = eta - (self.Q @ x_approx)

            # Accept or reject
            log_uniform = np.log(np.random.uniform())
            log_accept = np.amin([0.0, -residual.T @ (x - x_approx)])

            if not ar_step: log_accept = log_uniform + 1

            if log_uniform < log_accept:
                x = x_approx
                n_accepted += 1
            else:
                pass

            if keep_trace: trace[j,:] = x

        # Compute stats
        acceptance_rate = n_accepted/n_samples
        avg_cg_its = tot_cg_its/n_samples

        data = {
            "final_point": x,
            "trace": trace,
            "acceptance_rate": acceptance_rate,
            "avg_cg_its": avg_cg_its,
        }

        return data
        



