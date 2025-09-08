"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    k, _ = mixture.mu.shape
    
    total_post = np.zeros((n, k))
    log_likelihood = 0
    
    for i in range(n):
        mask = (X[i, :] != 0)
        
        X_cu = X[i, mask]
        d_observed = mask.sum()
        
        # Precompute constants
        const = -0.5 * d_observed * np.log(2 * np.pi)
    
        # Reshape for broadcasting
        X_reshaped = X_cu[np.newaxis, :]  # (n, 1, d_observed)
        
        mu_reshaped = mixture.mu[:, mask]  # (1, K, d_observed)
        
        # Compute squared distances (n, K)
        squared_distances = np.sum((X_reshaped - mu_reshaped)**2, axis=1)
        
        # Compute log probabilities without normalization (n, K)
        log_probs = const - 0.5 * d_observed * np.log(mixture.var) - squared_distances / (2 * mixture.var)
        
        # Add log of mixture weights
        log_weighted_probs = log_probs + np.log(mixture.p + 1e-16) # f(u, j) in the document
        
        # Compute log of denominator using log-sum-exp for numerical stability
        log_denom = logsumexp(log_weighted_probs, axis=0, keepdims=True)
    
        # Compute posterior probabilities
        post = np.exp(log_weighted_probs - log_denom)
        
        total_post[i, :] = post
        
        # Compute total log-likelihood
        log_likelihood += log_denom
    
    
    return total_post, log_likelihood

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    n, d = X.shape
    _, K = post.shape

    n_hat_for_p = post.sum(axis=0)
    p = n_hat_for_p / n
    
    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        for l in range(d):
            
            
            mask = (X[:, l] != 0 )
            X_l_rated = X[mask, l]
            n_hat = post[mask, j].sum(axis=0)
            if n_hat >= 1:
                mu[j, l] = (post[mask, j] @ X_l_rated) / n_hat
            else:
                mu[j, l] = mixture.mu[j, l]
        
        sse_numerator = 0.0      # Running total for the numerator
        norm_denominator = 0.0   # Running total for the denominator
        
        for i in range(n):
            # Create a mask for this specific user's rated movies
            user_mask = (X[i, :] != 0)
            d_observed = user_mask.sum() # Number of movies this user rated
        
            if d_observed > 0:
                # Get the observed ratings and corresponding new means
                observed_ratings = X[i, user_mask]
                observed_means = mu[j, user_mask]
        
                # Calculate this user's contribution to the numerator
                sq_error = np.sum((observed_ratings - observed_means)**2)
                sse_numerator += post[i, j] * sq_error
        
                # Calculate this user's contribution to the denominator
                norm_denominator += post[i, j] * d_observed
        
        # Now calculate the variance for cluster j
        if norm_denominator > 0:
            var[j] = sse_numerator / norm_denominator
        else:
            # If cluster has no points, keep old variance
            var[j] = mixture.var[j]
        
        # Apply minimum variance constraint 
        var[j] = max(var[j], min_variance)
        
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    prev_ll = None
    ll = None
    while prev_ll is None or (ll - prev_ll > 1e-6 * np.abs(ll)):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, ll



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    post, ll = estep(X, mixture)
    newX = post @ mixture.mu
    
    return newX

def spherical_gaussian_pdf(x, mu, sigma_2):
    """
    Calculates probability density for spherical Gaussian (Σ = σ²I)
    
    Args:
        x: (1, d) or (d,) array - A single data point
        mu: (1, d) or (d,) array - Mean vector
        sigma_2: float - Variance (same for all dimensions)
        
    Returns:
        float: Probability density
    """
    # Ensure we have 1D arrays for vector operations
    x = np.squeeze(x)  # Removes singleton dimensions: (1, d) -> (d,)
    mu = np.squeeze(mu)  # (1, d) -> (d,)
    
    d = x.shape[0]  # Number of dimensions
    
    # Calculate squared Euclidean distance (scalar)
    squared_distance = np.sum((x - mu) ** 2)
    
    # Calculate probability density
    normalization = 1.0 / np.sqrt((2 * np.pi * sigma_2) ** d)
    exponential = np.exp(-squared_distance / (2 * sigma_2))
    
    return normalization * exponential