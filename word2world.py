
"""
Word2World Social Platform - STAP v4.0
Bridging Ideas Together

Multi-Modal Hyperbolic Embeddings for Peace-Building Social Networks

This platform implements the STAP v4.0 framework with four major enhancements:
1. Multi-Modal Mahalanobis Distance (text + image + network + engagement)
2. Hyperbolic Geometry with Contrastive Loss
3. Siamese Neural Network for Bridge-Aware Scoring
4. Secure token handling for production deployment

Key Features:
- Multi-modal user profile representation
- Covariance-aware distance metrics (Mahalanobis)
- Hyperbolic semantic space for hierarchical structure
- Data-driven bridge quality prediction
- Scalable architecture with HNSW indexing
- Secure Hugging Face token management

Authors: shu bai, xiaoyue cao, jinxing chen, yunzhou dai, vieux valcin, huiyi zhang
Date: October 22, 2025
Version: 4.0
License: MIT

Installation:
    pip install numpy scipy scikit-learn torch sentence-transformers transformers pillow requests networkx node2vec hnswlib

Usage:
    from word2world import STAPv4PreprocessingLayer, STAPv4Config
    
    # Initialize
    config = STAPv4Config()
    stap = STAPv4PreprocessingLayer(config)
    
    # Generate user coordinates
    coordinate, confidence = stap.generate_semantic_coordinate(
        user_id="user123",
        user_corpus=["post1", "post2"],
        engagement_patterns={},
        user_images=[],
        user_connections=[]
    )
    
    # Find bridges
    bridges = stap.find_bridges(user_id="user123", k=10)

Environment Variables:
    HF_TOKEN: Hugging Face API token (optional, for gated models)
    
    Set via:
        export HF_TOKEN="your_token_here"
    
    Or in Python:
        import os
        os.environ['HF_TOKEN'] = "your_token_here"
"""

### I screwed around and did not check that hnswlib was installed lmao

import os
import sys
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from sklearn.neighbors import NearestNeighbors
import hnswlib

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# NLP and embeddings
from sentence_transformers import SentenceTransformer

# CLIP for image embeddings
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
from io import BytesIO

# Node2Vec for network embeddings
import networkx as nx
try:
    from node2vec import Node2Vec as Node2VecModel
except ImportError:
    Node2VecModel = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Secure Token Handling
# ============================================================================

def get_hf_token() -> Optional[str]:
    """
    Safely retrieve Hugging Face token from environment variables or Colab secrets.
    This function is secure for GitHub as it never hardcodes tokens.
    
    Returns:
        HF token string or None if not available
    """
    # Try environment variable first (for local development)
    token = os.getenv('HF_TOKEN')
    if token:
        logger.info("HF token loaded from environment variable")
        return token
    
    # Try Colab secrets (for Google Colab)
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        logger.info("HF token loaded from Colab secrets")
        return token
    except ImportError:
        logger.debug("Not running in Colab environment")
    except userdata.SecretNotFoundError:
        logger.warning("HF_TOKEN secret not found in Colab")
    except Exception as e:
        logger.warning(f"Error accessing Colab secrets: {e}")
    
    logger.info("No HF token available - using public models only")
    return None

# Get token safely
HF_TOKEN = get_hf_token()

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class STAPv4Config:
    """Configuration for STAP v4.0 processor with multi-modal support"""

    # Embedding models
    text_embedding_model: str = 'all-MiniLM-L6-v2'
    clip_model: str = 'openai/clip-vit-base-patch32'

    # Multi-modal fusion weights
    alpha_text: float = 0.54
    alpha_image: float = 0.18
    alpha_network: float = 0.18
    alpha_engagement: float = 0.10

    # High-dimensional space
    use_mahalanobis: bool = True
    use_diagonal_covariance: bool = True
    covariance_epsilon: float = 1e-6

    # Low-dimensional space
    target_dim: int = 32
    use_hyperbolic: bool = True
    use_contrastive_loss: bool = True

    # Hyperbolic parameters
    hyperbolic_margin: float = 2.0
    contrastive_weight: float = 0.1
    boundary_regularization: float = 0.01
    boundary_threshold: float = 0.95

    # STAP optimization
    n_neighbors: int = 15
    learning_rate: float = 0.01
    n_epochs: int = 200
    negative_sample_rate: int = 5

    # Heavy-tailed distribution (for Euclidean fallback)
    a: float = 1.577
    b: float = 0.895

    # Siamese network
    use_siamese: bool = True
    siamese_hidden_dim: int = 128
    siamese_output_dim: int = 32
    context_dim: int = 20

    # CLIP parameters
    use_clip: bool = True
    clip_image_size: int = 224

    # Node2Vec parameters
    use_node2vec: bool = True
    node2vec_dimensions: int = 128
    node2vec_walk_length: int = 30
    node2vec_num_walks: int = 200
    node2vec_p: float = 1.0
    node2vec_q: float = 1.0
    node2vec_window_size: int = 10
    node2vec_workers: int = 4

    # General
    random_state: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# CLIP Encoder (Secure Version)
# ============================================================================

class CLIPEncoder:
    """
    CLIP encoder for image-text embeddings with secure token handling.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", 
                 device: str = None, hf_token: Optional[str] = None):
        """
        Initialize CLIP encoder securely.

        Args:
            model_name: CLIP model identifier
            device: Device for computation
            hf_token: Optional Hugging Face token (uses secure retrieval if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use provided token or safely retrieve one
        token_to_use = hf_token or HF_TOKEN
        
        # Load model with authentication if token available
        try:
            if token_to_use:
                self.model = CLIPModel.from_pretrained(
                    model_name, 
                    use_auth_token=token_to_use
                )
                self.processor = CLIPProcessor.from_pretrained(
                    model_name,
                    use_auth_token=token_to_use
                )
                logger.info(f"CLIP encoder initialized with authentication: {model_name}")
            else:
                self.model = CLIPModel.from_pretrained(model_name)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                logger.info(f"CLIP encoder initialized without authentication: {model_name}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize CLIP encoder: {e}")
            # Fallback to without authentication
            try:
                self.model = CLIPModel.from_pretrained(model_name)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                logger.info(f"CLIP encoder fallback initialized: {model_name}")
            except Exception as fallback_error:
                logger.error(f"CLIP encoder completely failed: {fallback_error}")
                raise fallback_error

        self.embedding_dim = 512  # CLIP ViT-B/32
        self.model.to(self.device)
        self.model.eval()

    def encode_images(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Encode images to embeddings.

        Args:
            images: List of images (URLs, PIL Images, or numpy arrays)

        Returns:
            Image embeddings array
        """
        if not images:
            return np.zeros((0, self.embedding_dim))

        processed_images = []
        for img in images:
            if isinstance(img, str):
                if img.startswith(('http://', 'https://')):
                    try:
                        response = requests.get(img, timeout=5)
                        img = Image.open(BytesIO(response.content))
                    except Exception as e:
                        logger.warning(f"Failed to load image from URL: {e}")
                        continue
                else:
                    try:
                        img = Image.open(img)
                    except Exception as e:
                        logger.warning(f"Failed to load image from path: {e}")
                        continue
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            processed_images.append(img)

        if not processed_images:
            return np.zeros((0, self.embedding_dim))

        with torch.no_grad():
            inputs = self.processor(images=processed_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_image_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy()

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings array
        """
        if not texts:
            return np.zeros((0, self.embedding_dim))

        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy()

# ============================================================================
# Node2Vec Encoder
# ============================================================================

class Node2VecEncoder:
    """
    Node2Vec encoder for network structure embeddings.
    """

    def __init__(self, config: STAPv4Config):
        """
        Initialize Node2Vec encoder.

        Args:
            config: STAP v4.0 configuration
        """
        self.config = config
        self.graph = nx.Graph()
        self.model = None
        self.embeddings = {}

        if Node2VecModel is None:
            logger.warning("node2vec package not installed. Network embeddings will be disabled.")

        logger.info("Node2Vec encoder initialized")

    def add_edge(self, user1: str, user2: str, weight: float = 1.0):
        """
        Add edge to the social graph.

        Args:
            user1: First user ID
            user2: Second user ID
            weight: Edge weight
        """
        self.graph.add_edge(user1, user2, weight=weight)

    def add_edges(self, edges: List[Tuple[str, str, float]]):
        """
        Add multiple edges to the graph.

        Args:
            edges: List of (user1, user2, weight) tuples
        """
        for user1, user2, weight in edges:
            self.add_edge(user1, user2, weight)

    def train(self):
        """
        Train Node2Vec model on the current graph.
        """
        if Node2VecModel is None:
            logger.warning("node2vec package not available")
            return

        if self.graph.number_of_nodes() < 2:
            logger.warning("Insufficient nodes for Node2Vec training")
            return

        logger.info(f"Training Node2Vec on graph with {self.graph.number_of_nodes()} nodes")

        node2vec = Node2VecModel(
            self.graph,
            dimensions=self.config.node2vec_dimensions,
            walk_length=self.config.node2vec_walk_length,
            num_walks=self.config.node2vec_num_walks,
            p=self.config.node2vec_p,
            q=self.config.node2vec_q,
            workers=self.config.node2vec_workers,
            quiet=True
        )

        self.model = node2vec.fit(
            window=self.config.node2vec_window_size,
            min_count=1,
            batch_words=4
        )

        # Cache embeddings
        for node in self.graph.nodes():
            try:
                self.embeddings[node] = self.model.wv[str(node)]
            except KeyError:
                self.embeddings[node] = np.zeros(self.config.node2vec_dimensions)

        logger.info(f"Node2Vec training complete: {len(self.embeddings)} embeddings")

    def get_embedding(self, user_id: str) -> np.ndarray:
        """
        Get Node2Vec embedding for a user.

        Args:
            user_id: User identifier

        Returns:
            Node2Vec embedding vector
        """
        if user_id in self.embeddings:
            return self.embeddings[user_id]

        # Return zero vector if not found
        return np.zeros(self.config.node2vec_dimensions)

    def compute_bridge_weight(self, user1: str, user2: str,
                            semantic_positions: Dict[str, np.ndarray]) -> float:
        """
        Compute bridge-aware edge weight based on semantic distance.

        Args:
            user1: First user ID
            user2: Second user ID
            semantic_positions: Dictionary of user semantic coordinates

        Returns:
            Bridge-aware weight
        """
        if user1 not in semantic_positions or user2 not in semantic_positions:
            return 1.0

        pos1 = semantic_positions[user1]
        pos2 = semantic_positions[user2]

        # Semantic distance (normalized)
        distance = np.linalg.norm(pos1 - pos2)

        # Bridge score: higher for medium distances (optimal bridge range [0.3, 0.5])
        bridge_score = np.exp(-((distance - 0.4) ** 2) / 0.1)

        # Apply bridge weight strength
        weight = 1.0 + bridge_score

        return float(weight)

# ============================================================================
# Multi-Modal Mahalanobis Distance (High-Dimensional Space)
# ============================================================================

class MultiModalMahalanobisDistance:
    """
    Implements multi-modal Mahalanobis distance for high-dimensional space.
    Combines text, image, network, and engagement embeddings with covariance-aware distance.
    """

    def __init__(self, config: STAPv4Config):
        """
        Initialize multi-modal Mahalanobis distance calculator.

        Args:
            config: STAP v4.0 configuration
        """
        self.config = config
        self.Sigma = None
        self.Sigma_inv = None
        self.L = None  # Cholesky factor
        self.mu = None
        self.is_fitted = False

        logger.info("MultiModalMahalanobisDistance initialized")

    def fit(self, X_multimodal: np.ndarray):
        """
        Compute covariance matrix from multi-modal embeddings.

        Args:
            X_multimodal: (N, d_h) multi-modal embeddings
        """
        N, d_h = X_multimodal.shape

        if N < 2:
            logger.warning("Insufficient data for covariance estimation")
            return

        # Compute mean
        self.mu = np.mean(X_multimodal, axis=0)

        # Compute covariance
        X_centered = X_multimodal - self.mu

        if self.config.use_diagonal_covariance:
            # Diagonal approximation for efficiency
            variances = np.var(X_centered, axis=0)
            self.Sigma = np.diag(variances + self.config.covariance_epsilon)
            self.Sigma_inv = np.diag(1.0 / (variances + self.config.covariance_epsilon))
            self.L = np.diag(np.sqrt(variances + self.config.covariance_epsilon))
        else:
            # Full covariance matrix
            self.Sigma = (X_centered.T @ X_centered) / (N - 1)
            self.Sigma += self.config.covariance_epsilon * np.eye(d_h)

            # Cholesky decomposition for efficient distance computation
            try:
                self.L = cholesky(self.Sigma, lower=True)
                self.Sigma_inv = np.linalg.inv(self.Sigma)
            except np.linalg.LinAlgError:
                logger.warning("Cholesky decomposition failed, using diagonal approximation")
                self.config.use_diagonal_covariance = True
                self.fit(X_multimodal)
                return

        self.is_fitted = True
        logger.info(f"Covariance matrix fitted: shape={self.Sigma.shape}, diagonal={self.config.use_diagonal_covariance}")

    def distance(self, x_i: np.ndarray, x_j: np.ndarray) -> float:
        """
        Compute Mahalanobis distance between two embeddings.

        Args:
            x_i: (d_h,) embedding for user i
            x_j: (d_h,) embedding for user j

        Returns:
            Mahalanobis distance
        """
        if not self.is_fitted:
            # Fallback to Euclidean distance
            return np.linalg.norm(x_i - x_j)

        diff = x_i - x_j

        if self.config.use_diagonal_covariance:
            # Efficient computation for diagonal covariance
            z = diff * np.sqrt(np.diag(self.Sigma_inv))
            return np.linalg.norm(z)
        else:
            # Full Mahalanobis distance using Cholesky factor
            z = solve_triangular(self.L, diff, lower=True)
            return np.linalg.norm(z)

    def probability(self, x_i: np.ndarray, x_j: np.ndarray, sigma_i: float) -> float:
        """
        Compute high-dimensional connection probability.

        Args:
            x_i: (d_h,) embedding for user i
            x_j: (d_h,) embedding for user j
            sigma_i: Local bandwidth for user i

        Returns:
            Connection probability
        """
        d = self.distance(x_i, x_j)
        return np.exp(-d**2 / (2 * sigma_i**2))

# ============================================================================
# Hyperbolic Geometry (Low-Dimensional Space)
# ============================================================================

class HyperbolicSpace:
    """
    Implements hyperbolic geometry operations in the Poincaré ball model.
    """

    def __init__(self, dim: int):
        """
        Initialize hyperbolic space.

        Args:
            dim: Dimension of Poincaré ball
        """
        self.dim = dim
        self.eps = 1e-7  # Numerical stability

    def project_to_ball(self, y: np.ndarray, max_norm: float = 0.99) -> np.ndarray:
        """
        Project point onto Poincaré ball.

        Args:
            y: Point to project
            max_norm: Maximum norm (< 1)

        Returns:
            Projected point
        """
        norm = np.linalg.norm(y)
        if norm >= 1.0:
            return (max_norm / (norm + self.eps)) * y
        return y

    def hyperbolic_distance(self, y_i: np.ndarray, y_j: np.ndarray) -> float:
        """
        Compute hyperbolic distance in Poincaré ball.

        Args:
            y_i: Point in Poincaré ball
            y_j: Point in Poincaré ball

        Returns:
            Hyperbolic distance
        """
        # Ensure points are in the ball
        y_i = self.project_to_ball(y_i)
        y_j = self.project_to_ball(y_j)

        # Compute squared Euclidean distance
        diff_squared = np.sum((y_i - y_j) ** 2)

        # Compute norms
        norm_i_squared = np.sum(y_i ** 2)
        norm_j_squared = np.sum(y_j ** 2)

        # Hyperbolic distance formula
        numerator = 2 * diff_squared
        denominator = (1 - norm_i_squared) * (1 - norm_j_squared) + self.eps

        distance = np.arccosh(1 + numerator / denominator)

        return distance

    def mobius_addition(self, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Möbius addition in Poincaré ball.

        Args:
            y: Point in Poincaré ball
            v: Vector to add

        Returns:
            Result of Möbius addition
        """
        y_norm_sq = np.sum(y ** 2)
        v_norm_sq = np.sum(v ** 2)
        y_dot_v = np.dot(y, v)

        numerator = (1 + 2 * y_dot_v + v_norm_sq) * y + (1 - y_norm_sq) * v
        denominator = 1 + 2 * y_dot_v + y_norm_sq * v_norm_sq + self.eps

        result = numerator / denominator
        return self.project_to_ball(result)

    def exponential_map(self, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map at point y in direction v.

        Args:
            y: Base point in Poincaré ball
            v: Tangent vector

        Returns:
            Result of exponential map
        """
        v_norm = np.linalg.norm(v)

        if v_norm < self.eps:
            return y

        # Hyperbolic norm
        y_norm_sq = np.sum(y ** 2)
        lambda_y = 2 / (1 - y_norm_sq + self.eps)

        # Exponential map formula
        v_normalized = v / (v_norm + self.eps)
        tanh_term = np.tanh(lambda_y * v_norm / 2)

        result = self.mobius_addition(y, tanh_term * v_normalized)
        return result

# ============================================================================
# Contrastive Loss for Hyperbolic Space
# ============================================================================

class HyperbolicContrastiveLoss:
    """
    Implements contrastive loss in hyperbolic space.
    """

    def __init__(self, hyperbolic_space: HyperbolicSpace, margin: float = 2.0):
        """
        Initialize contrastive loss.

        Args:
            hyperbolic_space: HyperbolicSpace instance
            margin: Margin for dissimilar pairs
        """
        self.hyperbolic_space = hyperbolic_space
        self.margin = margin

    def compute_loss(
        self,
        Y: np.ndarray,
        similar_pairs: List[Tuple[int, int]],
        dissimilar_pairs: List[Tuple[int, int]]
    ) -> float:
        """
        Compute contrastive loss.

        Args:
            Y: (N, d) low-dimensional embeddings
            similar_pairs: List of (i, j) similar pairs
            dissimilar_pairs: List of (i, k) dissimilar pairs

        Returns:
            Contrastive loss value
        """
        loss_similar = 0.0
        for i, j in similar_pairs:
            d_H = self.hyperbolic_space.hyperbolic_distance(Y[i], Y[j])
            loss_similar += d_H ** 2

        loss_dissimilar = 0.0
        for i, k in dissimilar_pairs:
            d_H = self.hyperbolic_space.hyperbolic_distance(Y[i], Y[k])
            loss_dissimilar += max(0, self.margin - d_H) ** 2

        total_loss = loss_similar / max(len(similar_pairs), 1) + \
                    loss_dissimilar / max(len(dissimilar_pairs), 1)

        return total_loss

    def compute_gradient(
        self,
        Y: np.ndarray,
        similar_pairs: List[Tuple[int, int]],
        dissimilar_pairs: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute gradient of contrastive loss.

        Args:
            Y: (N, d) low-dimensional embeddings
            similar_pairs: List of similar pairs
            dissimilar_pairs: List of dissimilar pairs

        Returns:
            Gradient array (N, d)
        """
        N, d = Y.shape
        grad = np.zeros((N, d))

        # Similar pairs gradient
        for i, j in similar_pairs:
            d_H = self.hyperbolic_space.hyperbolic_distance(Y[i], Y[j])
            if d_H > 1e-10:
                coeff = 2 * d_H
                grad[i] += coeff * (Y[i] - Y[j])
                grad[j] += coeff * (Y[j] - Y[i])

        # Dissimilar pairs gradient
        for i, k in dissimilar_pairs:
            d_H = self.hyperbolic_space.hyperbolic_distance(Y[i], Y[k])
            if d_H < self.margin:
                coeff = -2 * (self.margin - d_H)
                grad[i] += coeff * (Y[i] - Y[k])
                grad[k] += coeff * (Y[k] - Y[i])

        # Normalize by number of pairs
        grad /= max(len(similar_pairs) + len(dissimilar_pairs), 1)

        return grad

# ============================================================================
# Siamese Neural Network for Bridge-Aware Scoring
# ============================================================================

class SiameseNetwork(nn.Module):
    """
    Siamese neural network for learning bridge success probability.
    """

    def __init__(self, d_low: int, d_context: int, d_hidden: int, d_output: int):
        """
        Initialize Siamese network.

        Args:
            d_low: Dimension of low-dimensional embeddings
            d_context: Dimension of context vectors
            d_hidden: Hidden layer dimension
            d_output: Output dimension
        """
        super(SiameseNetwork, self).__init__()

        # Input dimension: low-dim embedding + context
        d_input = d_low + d_context

        # Shared tower
        self.tower = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, d_output)
        )

        # Bridge prediction head
        self.bridge_head = nn.Sequential(
            nn.Linear(d_output * 2, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, y_i: torch.Tensor, context_i: torch.Tensor,
                y_j: torch.Tensor, context_j: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            y_i: Low-dimensional embedding for user i (batch_size, d_low)
            context_i: Context vector for user i (batch_size, d_context)
            y_j: Low-dimensional embedding for user j (batch_size, d_low)
            context_j: Context vector for user j (batch_size, d_context)

        Returns:
            Bridge probability (batch_size, 1)
        """
        # Concatenate embeddings and context
        x_i = torch.cat([y_i, context_i], dim=1)
        x_j = torch.cat([y_j, context_j], dim=1)

        # Pass through shared tower
        h_i = self.tower(x_i)
        h_j = self.tower(x_j)

        # Concatenate representations
        h_pair = torch.cat([h_i, h_j], dim=1)

        # Predict bridge probability
        bridge_prob = self.bridge_head(h_pair)

        return bridge_prob

    def bridge_score(self, y_i: np.ndarray, context_i: np.ndarray,
                    y_j: np.ndarray, context_j: np.ndarray,
                    device: str = 'cpu') -> float:
        """
        Compute bridge score for a single pair.

        Args:
            y_i: Low-dimensional embedding for user i
            context_i: Context vector for user i
            y_j: Low-dimensional embedding for user j
            context_j: Context vector for user j
            device: Device for computation

        Returns:
            Bridge score
        """
        self.eval()
        with torch.no_grad():
            y_i_t = torch.from_numpy(y_i).float().unsqueeze(0).to(device)
            context_i_t = torch.from_numpy(context_i).float().unsqueeze(0).to(device)
            y_j_t = torch.from_numpy(y_j).float().unsqueeze(0).to(device)
            context_j_t = torch.from_numpy(context_j).float().unsqueeze(0).to(device)

            score = self.forward(y_i_t, context_i_t, y_j_t, context_j_t)

        return float(score.item())

# ============================================================================
# STAP v4.0 Preprocessing Layer (GitHub-Friendly Version)
# ============================================================================

class STAPv4PreprocessingLayer:
    """
    STAP v4.0 Preprocessing Layer with integrated multi-modal support:
    1. Multi-Modal Mahalanobis Distance (high-dimensional)
    2. Hyperbolic Geometry with Contrastive Loss (low-dimensional)
    3. Siamese Neural Network (bridge-aware scoring)
    4. CLIP Integration (image-text embeddings)
    5. Node2Vec Integration (network structure embeddings)
    
    GitHub-Friendly: No hardcoded secrets, secure token handling
    """

    def __init__(self, config: Optional[STAPv4Config] = None):
        """
        Initialize STAP v4.0 preprocessing layer securely.

        Args:
            config: STAP v4.0 configuration
        """
        self.config = config or STAPv4Config()

        # Initialize text embedding model
        logger.info(f"Loading text embedding model: {self.config.text_embedding_model}")
        self.text_embedding_model = SentenceTransformer(self.config.text_embedding_model)
        self.text_embedding_dim = self.text_embedding_model.get_sentence_embedding_dimension()

        # Initialize CLIP encoder with secure token handling
        self.clip_encoder = None
        if self.config.use_clip:
            try:
                self.clip_encoder = CLIPEncoder(
                    self.config.clip_model, 
                    self.config.device,
                    hf_token=HF_TOKEN  # Pass the securely retrieved token
                )
                self.image_embedding_dim = self.clip_encoder.embedding_dim
            except Exception as e:
                logger.warning(f"Failed to initialize CLIP encoder: {e}")
                self.config.use_clip = False
                self.image_embedding_dim = 512
        else:
            self.image_embedding_dim = 512

        # Initialize Node2Vec encoder
        self.node2vec_encoder = None
        if self.config.use_node2vec:
            self.node2vec_encoder = Node2VecEncoder(self.config)
            self.network_embedding_dim = self.config.node2vec_dimensions
        else:
            self.network_embedding_dim = 128

        # Engagement embedding dimension
        self.engagement_embedding_dim = 32

        # Total high-dimensional size
        self.high_dim_size = (
            self.text_embedding_dim +
            self.image_embedding_dim +
            self.network_embedding_dim +
            self.engagement_embedding_dim
        )

        # Storage for global state
        self.high_dim_embeddings = []  # Multi-modal high-dimensional embeddings
        self.low_dim_embeddings = []   # Hyperbolic low-dimensional embeddings
        self.user_id_map = {}
        self.index_to_user_id = {}

        # Initialize components
        self.mahalanobis_distance = MultiModalMahalanobisDistance(self.config)
        self.hyperbolic_space = HyperbolicSpace(self.config.target_dim)
        self.contrastive_loss = HyperbolicContrastiveLoss(
            self.hyperbolic_space,
            margin=self.config.hyperbolic_margin
        )

        # Siamese network (will be loaded if available)
        self.siamese_network = None
        if self.config.use_siamese:
            self.siamese_network = SiameseNetwork(
                d_low=self.config.target_dim,
                d_context=self.config.context_dim,
                d_hidden=self.config.siamese_hidden_dim,
                d_output=self.config.siamese_output_dim
            ).to(self.config.device)

        # HNSW index
        self.hnsw_index = None
        self.is_fitted = False

        # Cache
        self.P_high = None
        self.local_sigmas = None

        logger.info(f"STAP v4.0 initialized: {self.high_dim_size}D -> {self.config.target_dim}D (hyperbolic={self.config.use_hyperbolic})")

    def generate_semantic_coordinate(
        self,
        user_id: str,
        user_corpus: List[str],
        engagement_patterns: Dict,
        image_data: Optional[List] = None,
        network_connections: Optional[List[Tuple[str, str, float]]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Generate semantic coordinate using STAP v4.0 framework.

        Args:
            user_id: Unique user identifier
            user_corpus: List of user's text content
            engagement_patterns: Dict with interaction data (likes, shares, comments, etc.)
            image_data: Optional list of images (URLs, PIL Images, or numpy arrays)
            network_connections: Optional list of (user1, user2, weight) tuples

        Returns:
            Tuple of (semantic_coordinate, confidence_score)
        """
        # Step 1: Generate multi-modal high-dimensional embedding
        multimodal_embedding = self._generate_multimodal_embedding(
            user_corpus,
            engagement_patterns,
            image_data,
            network_connections
        )

        # Step 2: Add to global collection
        if user_id in self.user_id_map:
            idx = self.user_id_map[user_id]
            self.high_dim_embeddings[idx] = multimodal_embedding
        else:
            idx = len(self.high_dim_embeddings)
            self.high_dim_embeddings.append(multimodal_embedding)
            self.user_id_map[user_id] = idx
            self.index_to_user_id[idx] = user_id

            # Initialize low-dimensional embedding in Poincaré ball
            initial_embedding = np.random.randn(self.config.target_dim) * 0.01
            initial_embedding = self.hyperbolic_space.project_to_ball(initial_embedding)
            self.low_dim_embeddings.append(initial_embedding)

        # Step 3: Update Node2Vec graph if network connections provided
        if network_connections and self.node2vec_encoder:
            self.node2vec_encoder.add_edges(network_connections)

        # Step 4: Refit STAP if enough users
        if len(self.high_dim_embeddings) >= self.config.n_neighbors:
            if not self.is_fitted or len(self.high_dim_embeddings) % 10 == 0:
                self._fit_stap_v4_projection()

        # Step 5: Get optimized coordinate
        semantic_coordinate = self.low_dim_embeddings[idx]

        # Step 6: Calculate confidence
        confidence = self._calculate_confidence(
            user_corpus,
            engagement_patterns
        )

        # Step 7: Update HNSW index
        if self.is_fitted:
            self._update_hnsw_index()

        return semantic_coordinate, confidence

    def _generate_multimodal_embedding(
        self,
        text_corpus: List[str],
        engagement_patterns: Dict,
        image_data: Optional[List],
        network_connections: Optional[List[Tuple[str, str, float]]]
    ) -> np.ndarray:
        """
        Generate multi-modal embedding by fusing text, image, network, and engagement data.

        Args:
            text_corpus: List of text strings
            engagement_patterns: Dictionary of engagement metrics
            image_data: Optional image data
            network_connections: Optional network connections

        Returns:
            Multi-modal embedding vector
        """
        # Text embedding
        text_emb = self._generate_text_embeddings(text_corpus)

        # Image embedding using CLIP
        if image_data is not None and len(image_data) > 0 and self.clip_encoder:
            image_embeddings = self.clip_encoder.encode_images(image_data)
            if len(image_embeddings) > 0:
                image_emb = np.mean(image_embeddings, axis=0)
            else:
                image_emb = np.zeros(self.image_embedding_dim)
        else:
            image_emb = np.zeros(self.image_embedding_dim)

        # Network embedding using Node2Vec
        # Note: This requires the user to be in the graph and the model to be trained
        network_emb = np.zeros(self.network_embedding_dim)

        # Engagement embedding
        engagement_emb = self._generate_engagement_embedding(engagement_patterns)

        # Weighted concatenation
        multimodal_emb = np.con
