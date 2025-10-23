"""
Robust Covariance Estimation Module for Word2World
===================================================

Implements robust low-rank + diagonal covariance estimation using:
- Minimum Covariance Determinant (MCD) for outlier resistance
- SVD-based low-rank approximation for scalability
- Cholesky decomposition for efficient distance computation

Mathematical Framework (Section 2.1.2):
- Σ_robust = U_k U_k^T + D
- where U_k contains top-k singular vectors
- D is diagonal residual
- Cholesky: Σ = LL^T for efficient triangular solve

Authors: Shu Bai, Xiaoyue Cao, Jinxing Chen, Yunzhou Dai, Vieux Valcin, Huiyi Zhang
Version: 4.0
Date: October 22, 2025
License: MIT
"""

import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
import logging
from sklearn.covariance import MinCovDet
from scipy.linalg import solve_triangular, cholesky

logger = logging.getLogger(__name__)


@dataclass
class CovarianceConfig:
    """Configuration for covariance estimation strategies."""
    
    method: Literal['classical', 'robust', 'cellwise'] = 'robust'
    rank: Optional[int] = None  # Auto: ceil(sqrt(d))
    epsilon: float = 1e-6
    mcd_support_fraction: Optional[float] = None  # Auto-determined by MCD
    
    def get_rank(self, n_features: int) -> int:
        """Get rank for low-rank approximation."""
        if self.rank is not None:
            return min(self.rank, n_features)
        return max(1, int(np.ceil(np.sqrt(n_features))))


class RobustCovarianceEstimator:
    """
    Robust covariance estimator using MCD with low-rank + diagonal decomposition.
    
    Implements Section 2.1.2 of the technical paper:
    - Minimum Covariance Determinant for outlier resistance
    - Low-rank approximation via SVD for scalability
    - Diagonal residual for full-rank representation
    - Cholesky decomposition for efficient distance computation
    """
    
    def __init__(self, config: CovarianceConfig):
        """
        Initialize robust covariance estimator.
        
        Args:
            config: Covariance estimation configuration
        """
        self.config = config
        self.cov = None
        self.L = None
        self.mu = None
        self.rank = None
        self.U_k = None
        self.D = None
        self.mcd_estimator = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'RobustCovarianceEstimator':
        """
        Fit robust covariance estimator.
        
        Args:
            X: (N, d) data matrix
            
        Returns:
            self
        """
        N, d = X.shape
        
        if N < 2:
            logger.warning("Insufficient samples for covariance estimation")
            return self
            
        self.rank = self.config.get_rank(d)
        
        if self.config.method == 'classical':
            self._fit_classical(X)
        elif self.config.method == 'robust':
            self._fit_robust(X)
        elif self.config.method == 'cellwise':
            self._fit_cellwise(X)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
            
        self.is_fitted = True
        return self
        
    def _fit_classical(self, X: np.ndarray):
        """Classical sample covariance with low-rank decomposition."""
        self.mu = np.mean(X, axis=0)
        X_centered = X - self.mu
        
        # Sample covariance
        cov_full = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        
        # Low-rank decomposition
        self._decompose_covariance(cov_full)
        
    def _fit_robust(self, X: np.ndarray):
        """Robust covariance using MCD with low-rank decomposition."""
        # MCD estimation
        mcd = MinCovDet(
            support_fraction=self.config.mcd_support_fraction,
            random_state=42
        )
        
        try:
            mcd.fit(X)
            self.mcd_estimator = mcd
            self.mu = mcd.location_
            cov_robust = mcd.covariance_
            
            # Low-rank decomposition
            self._decompose_covariance(cov_robust)
            
            logger.info(f"MCD fitted: support={mcd.support_.sum()}/{X.shape[0]}")
            
        except Exception as e:
            logger.warning(f"MCD failed: {e}, falling back to classical")
            self._fit_classical(X)
            
    def _fit_cellwise(self, X: np.ndarray):
        """Cellwise robust covariance (placeholder for RobPy integration)."""
        logger.warning("Cellwise MCD not implemented, using robust MCD")
        self._fit_robust(X)
        
    def _decompose_covariance(self, cov_full: np.ndarray):
        """
        Decompose covariance into low-rank + diagonal.
        
        Σ ≈ U_k U_k^T + D
        
        Args:
            cov_full: Full covariance matrix
        """
        d = cov_full.shape[0]
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(cov_full)
        
        # Low-rank factor: U_k * sqrt(S_k)
        self.U_k = U[:, :self.rank] * np.sqrt(S[:self.rank])
        
        # Diagonal residual
        residual = cov_full - self.U_k @ self.U_k.T
        self.D = np.diag(np.diag(residual))
        
        # Reconstruct approximation
        self.cov = self.U_k @ self.U_k.T + self.D
        
        # Add regularization
        self.cov += self.config.epsilon * np.eye(d)
        
        # Cholesky decomposition
        try:
            self.L = cholesky(self.cov, lower=True)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed, increasing regularization")
            self.cov += 10 * self.config.epsilon * np.eye(d)
            self.L = cholesky(self.cov, lower=True)
            
        logger.info(f"Covariance decomposed: rank={self.rank}, dim={d}")
        
    def mahalanobis_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mahalanobis distance using Cholesky factor.
        
        d_M(x, y) = sqrt((x-y)^T Σ^{-1} (x-y))
                  = ||L^{-1}(x-y)||_2
        
        Args:
            x: First vector
            y: Second vector
            
        Returns:
            Mahalanobis distance
        """
        if not self.is_fitted:
            return np.linalg.norm(x - y)
            
        diff = np.asarray(x) - np.asarray(y)
        z = solve_triangular(self.L, diff, lower=True)
        return np.sqrt(np.dot(z, z))
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to whitened space: Z = L^{-1}(X - μ).
        
        Args:
            X: (N, d) data matrix
            
        Returns:
            (N, d) whitened data
        """
        if not self.is_fitted:
            return X
            
        X_centered = X - self.mu
        return solve_triangular(self.L, X_centered.T, lower=True).T
        
    def get_covariance(self) -> np.ndarray:
        """Get estimated covariance matrix."""
        return self.cov if self.is_fitted else None
        
    def get_precision(self) -> np.ndarray:
        """Get precision matrix (inverse covariance)."""
        if not self.is_fitted:
            return None
        return np.linalg.inv(self.cov)
