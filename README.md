# Word2World
**Bridging Ideas Together**

[![Version](https://img.shields.io/badge/version-4.0-blue.svg)](https://github.com/word2world/stap) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/) [![Projects for Peace](https://img.shields.io/badge/Projects%20for%20Peace-2025-orange.svg)](https://www.projectsforpeace.org/)

A $10,000 initiative funded by the *A Kathryn W. Davis Projects for Peace* by 
Translation, Interpretation, and Localization Management (TILM) students at the
Middlebury Institute of International Studies at Monterey (MIIS)

**Authors:** Shu Bai, Xiaoyue Cao, Jinxing Chen, Yunzhou Dai, Vieux Valcin, Huiyi Zhang
**Roles:**
Technical Lead: Valcin, Vieux
Engineer, Front-end: Chen, Jinxing
Engineer, Back-end: Cao, Xiaoyue
Project Manager: Bai, Shu
Project Administrator: Zhang, Huiyi
Project Administrator: Dai, Yunzhou 

---

## Overview

**Word2World** is a next-generation platform that leverages advanced mathematical frameworks to bridge ideological divides and foster constructive dialogue. Developed in part as a **Kathryn W. Davis Projects for Peace** initiative in partnership with the Middlebury Institute of International Studies at Monterey, Word2World represents a paradigm shift from engagement-driven algorithms to a promising peace-building technological solution.

With the **STAP v4.0 (Semantic Topology-Aware Projection)** framework at its core, which amalgamates multi-modal embeddings, hyperbolic geometry, manifold-aware regularization, and deep learning to identify "bridge" content that connects users with different perspectives.

### The Problem We're Solving

In 2017, Facebook's (and other social media) algorithms amplified hate speech that contributed to genocide in Myanmar [1]. In 2021, social media platforms mobilized a violent insurrection at the U.S. Capitol. During the COVID-19 pandemic, algorithmic amplification of health misinformation (i.e., "infodemics") led to at least 800 documented deaths [2]. The pattern is clear and devastating: **social media platforms designed to maximize engagement systematically amplify conflict, polarization, and violence.**

Most responses to this crisis are reactive—better content moderation, fact-checking, user education, etc. These are necessary but insufficient. They treat the symptoms, not the disease. The fundamental problem is structural: social media algorithms are optimized for engagement, and divisive content generates more engagement than bridge-building dialogue.

### Our Solution

It is a platform designed from the ground up for peace-building, not engagement. 

### What Makes STAP v4.0 Unique?

Unlike traditional recommendation systems that create echo chambers, our native STAP v4.0 architecture actively identifies and promotes content that:
- **Validates** your existing views (Neighbors)
- **Expands** your perspective with related but different ideas (Bridges)
- **Challenges** you with distant viewpoints (Horizons)

The system uses **five (5) groundbreaking mathematical developments**:

1. **Multi-Modal Mahalanobis Distance** - Combines text, images, network, and engagement data with robust covariance estimation (MCD) and low-rank approximation

2. **Hyperbolic Geometry** - Represents semantic hierarchies in curved Poincaré ball space, naturally encoding ideological trees

3. **Manifold-Aware Regularization** - Five regularizers ensure stable optimization: curvature-sensitive gradients, Taylor consistency, conformality, radius margin, and angular entropy

4. **Contrastive Learning** - MoCo (Momentum Contrast) with hard negative mining for robust boundary detection

5. **Siamese Neural Networks** - Learns to predict bridge success from historical interaction data with attention mechanisms

---

## Table of Contents

- [Projects for Peace Mission](#projects-for-peace-mission)
- [Features](#features)
- [Mathematical Framework](#mathematical-framework)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Deployment](#deployment)
- [Research](#research)
- [Contributing](#contributing)
- [License](#license)

---

## Projects for Peace Mission

### Building the Digital Infrastructure for Peace

The current state of social media is not inevitable. The algorithms that amplify hate, the echo chambers that deepen division, and the viral spread of misinformation are all the result of design choices—choices that prioritized engagement over understanding, and profit over peace.

**Word2World demonstrates that different choices are possible.**

We have built the foundation of a platform that identifies and promotes common ground. We have implemented algorithms that bridge divides rather than deepen them. We have created a system that learns and improves continuously based on user feedback.

### How Word2World Promotes Peace

Traditional social media creates echo chambers by showing users content that confirms their existing beliefs. This deepens polarization. Word2World systematically exposes users to perspectives that are different enough to expand their worldview, but similar enough to be comprehensible and non-threatening.

Research in conflict resolution and social psychology supports this approach:

1. **Contact Hypothesis**: Exposure to out-group members under the right conditions reduces prejudice (Allport, 1954)

2. **Common Ground Theory**: Finding shared values and interests is the foundation of dialogue (Clark, 1996)

3. **Perspective-Taking**: Understanding others' viewpoints increases empathy and reduces hostility (Galinsky & Moskowitz, 2000)

STAP operationalizes these principles at scale. By identifying and promoting content that bridges divides, Word2World creates the conditions for constructive dialogue and mutual understanding.

### Why This Project Will Succeed

1. **Proven Technology**: We have built a sophisticated, functional prototype with over **3,000 lines** of production-quality code implementing STAP v4.0

2. **Rigorous Evaluation**: We will use validated instruments, statistical analysis, and mixed-methods research to provide concrete evidence of impact

3. **Scalable Design**: The architecture is designed from the ground up for global deployment, handling 1M+ users

4. **Expert Team**: Our team combines technical expertise (AI development, localization engineering, applied engineering), domain knowledge (conflict resolution, peace studies), and project management skills

5. **Institutional Support**: MIIS provides an ideal environment with a globally-minded community and strong institutional commitment

6. **Urgent Need**: The evidence of social media's role in violence is overwhelming. There is a clear and pressing need for alternative models

7. **Replicable Model**: The "Digital Peace-Building Toolkit" will enable other institutions and organizations to adopt Word2World, multiplying impact

### MIIS Pilot Deployment

The MIIS pilot will:

- Deploy Word2World for local MIIS assessment  (500+ students, faculty, staff)
- Measure impact on dialogue quality, perspective-taking, and community cohesion
- Validate the STAP framework in a real-world setting
- Create a replicable model for other organizations and institutions
- Produce a comprehensive evaluation report and "Digital Peace-Building Toolkit"

---

## Features

### Core Capabilities

**Multi-Modal User Representation**
- Text embeddings via Sentence-BERT (384D)
- Image embeddings via CLIP (512D)
- Network embeddings via Node2Vec (128D)
- Engagement embeddings via Behavioral Aggregation (32D)
- Weighted fusion: α_text=0.54, α_image=0.18, α_network=0.18, α_engagement=0.10

**Advanced Distance Metrics**
- **Robust Mahalanobis distance** with MCD (Minimum Covariance Determinant)
- **Low-rank + diagonal approximation** for 97% memory reduction
- **Hyperbolic distance** in Poincaré ball with Taylor approximation
- **Contrastive loss** with MoCo for boundary classification
- **Confidence scoring** for coordinate reliability

**Manifold-Aware Regularization (New in v4.0)**
- **Curvature-sensitive gradient regularizer** (L_curv): Prevents exploding gradients near boundary
- **Taylor consistency penalty** (L_jac): Maintains geometric fidelity
- **Manifold conformality regularizer** (L_conf): Preserves conformal structure
- **Contrastive radius margin** (L_rad): Enhances hierarchical clustering
- **Manifold entropy regularizer** (L_entropy): Prevents mode collapse

**Advanced Contrastive Learning (New in v4.0)**
- **MoCo (Momentum Contrast)** with queue-based negative sampling
- **Hard negative mining** for robust boundary detection
- **Hyperbolic triplet loss** for hierarchical embeddings
- **Attention-based Siamese networks** for bridge prediction

**Intelligent Recommendation System**
- Three-tier recommendations: Neighbors, Bridges, Horizons
- Bridge-aware scoring via Siamese neural networks
- Real-time updates with incremental learning
- O(log n) nearest neighbor search via HNSW

**Scalable Architecture**
- Handles 1M+ users
- Incremental covariance updates
- Efficient diagonal approximation option
- GPU acceleration for Siamese inference
- Configurable precision/performance tradeoffs

**Research-Backed Design**
- Addresses pluralistic ignorance
- Mitigates spiral of silence
- Promotes intellectual humility
- Reduces polarization

---

## Mathematical Framework

### High-Dimensional Space: Multi-Modal Mahalanobis Distance

**Objective:** Capture semantic similarity across multiple modalities while accounting for feature correlations.

**Multi-Modal Fusion:**

```
x_i^(multimodal) = [α_text · x_i^(text), α_image · x_i^(image), α_network · x_i^(network), α_engagement · x_i^(engagement)]
```

**Mahalanobis Distance:**

```
d_M(x_i, x_j) = √((x_i - x_j)ᵀ Σ⁻¹ (x_i - x_j))
```

where:
- `Σ` is the **robust covariance matrix** estimated via Minimum Covariance Determinant (MCD)
- Low-rank approximation: `Σ ≈ U_k U_kᵀ + D` (achieves 97% memory reduction)
- `Σ⁻¹` is computed via Cholesky decomposition for efficiency
- Diagonal approximation available for large-scale systems

**High-Dimensional Connection Probability:**

```
P(i,j) = exp(-d_M(x_i, x_j)² / (2σ_i²))
```

where `σ_i` is locally adaptive bandwidth determined by binary search to achieve target perplexity.

---

### Low-Dimensional Space: Hyperbolic Geometry

**Objective:** Embed users in a curved space that naturally represents hierarchical semantic structure.

**Poincaré Ball Model:**

```
B^d = {y ∈ ℝ^d : ||y|| < 1}
```

**Hyperbolic Distance:**

```
d_H(y_i, y_j) = arcosh(1 + 2||y_i - y_j||² / ((1 - ||y_i||²)(1 - ||y_j||²)))
```

**Properties:**
- Exponential volume growth → captures hierarchies
- Moderate views near center (||y|| ≈ 0)
- Extreme views near boundary (||y|| → 1)
- Natural tree structure for ideological space

**Low-Dimensional Connection Probability:**

```
Q(i,j) = 1 / (1 + a·d_H(y_i, y_j)^(2b))
```

where `a = 1.577` and `b = 0.895` define a heavy-tailed distribution for better separation.

---

### Robust Covariance Estimation (v4.0)

**Objective:** Provide outlier-resistant distance computation for noisy social media data.

**Minimum Covariance Determinant (MCD):**

```
H* = argmin_{H⊆{1,...,N}, |H|=h} det(Σ_H)
```

where `h = ⌊(N+d+1)/2⌋` is the minimum support fraction.

**Low-Rank Decomposition:**

```
Σ_MCD = U Λ Uᵀ  (via SVD)
Σ ≈ U_k U_kᵀ + D  (low-rank + diagonal)
```

where `k = ⌈√d⌉` (default) balances accuracy and efficiency.

**Benefits:**
- **Outlier Resistance**: Automatically downweights spam, bots, extreme users
- **Memory Efficiency**: 97% reduction (8.9 MB → 0.27 MB for d=1056)
- **Computational Speed**: 30× faster distance computation vs full covariance
- **Scalability**: Handles 1M+ users without degradation

---

### Manifold-Aware Regularization (v4.0)

**Integrated Objective Function:**

```
L_total = L_STAP + α·L_contrast + β·L_boundary + γ(L_curv + L_jac + L_conf + L_rad + L_entropy)
```

**Five Regularizers:**

1. **Curvature-Sensitive Gradient (L_curv)**
   ```
   L_curv = (1/N) Σ_i ‖∇_y_i L_STAP‖² · (1 - ‖y_i‖²)⁻²
   ```
   Prevents exploding gradients near boundary via metric tensor scaling.

2. **Taylor Consistency Penalty (L_jac)**
   ```
   L_jac = (1/N²) Σ_i Σ_j ‖d_Hyperbolic(y_i, y_j) - d_Taylor(y_i, y_j)‖²
   ```
   Maintains geometric fidelity with approximate distance calculations.

3. **Manifold Conformality (L_conf)**
   ```
   L_conf = (1/N) Σ_i ‖y_i‖² · ‖∇_y_i L_STAP‖²
   ```
   Preserves conformal structure of Poincaré ball.

4. **Contrastive Radius Margin (L_rad)**
   ```
   L_rad = (1/|P|) Σ_(i,j)∈P max(0, ‖y_i‖ - ‖y_j‖ + m_rad)
   ```
   Enhances hierarchical clustering via radial positioning.

5. **Manifold Entropy (L_entropy)**
   ```
   L_entropy = -(1/N) Σ_i Σ_k p_k(y_i) log p_k(y_i)
   ```
   Prevents mode collapse, ensures diverse angular distribution.

**Hyperparameters:** α = 0.5, β = 0.3, γ = 0.1, with individual weights w_curv = 0.3, w_jac = 0.25, w_conf = 0.2, w_rad = 0.15, w_entropy = 0.1.

---

### Contrastive Learning (v4.0)

**MoCo (Momentum Contrast):**

```
L_MoCo = -log(exp(q·k+ / τ) / (exp(q·k+ / τ) + Σ_k- exp(q·k- / τ)))
```

where:
- `q` = query embedding
- `k+` = positive key (similar user)
- `k-` = negative keys from queue (dissimilar users)
- `τ` = temperature parameter

**Benefits:**
- Large, consistent dictionary of negatives (K=4096)
- Momentum encoder for stable keys
- Hard negative mining for robust boundaries

**Hyperbolic Triplet Loss:**

```
L_triplet = max(0, d_H(anchor, positive) - d_H(anchor, negative) + margin)
```

Enforces hierarchical structure in hyperbolic space.

---

### Siamese Neural Network for Bridge Prediction

**Architecture:**

```
Input: [y_i, y_j, context_vector] → MLP → P(bridge_success)
```

**Context Vector (32D):**
- User engagement patterns (8D)
- Sentiment alignment (4D)
- Openness to new ideas (4D)
- Historical bridge acceptance (4D)
- Topic diversity (4D)
- Interaction history (8D)

**Attention Mechanism (New in v4.0):**

```
attention_weights = softmax(W_q·y_i ⊙ W_k·y_j)
attended_features = attention_weights ⊙ concat(y_i, y_j)
```

Focuses on semantically relevant dimensions for bridge prediction.

**Training:**
- Positive examples: Accepted bridge recommendations
- Negative examples: Rejected or ignored recommendations
- Loss: Binary cross-entropy
- Optimizer: Adam with learning rate scheduling

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (optional, for GPU acceleration)
- Hugging Face account (for model access)

### Install Dependencies

```bash
pip install torch transformers sentence-transformers scikit-learn networkx gensim nltk numpy pandas matplotlib seaborn plotly
```

### Environment Setup

Set your Hugging Face token:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or in Python:

```python
import os
os.environ['HF_TOKEN'] = "your_huggingface_token_here"
```

---

## Quick Start

### Basic Usage

```python
from word2world import STAPv4PreprocessingLayer, STAPv4Config

# Initialize with default configuration
config = STAPv4Config()
stap = STAPv4PreprocessingLayer(config)

# Generate semantic coordinate for a user
user_corpus = [
    "Climate change requires urgent global action",
    "Renewable energy is the path forward",
    "We must transition away from fossil fuels"
]

engagement_patterns = {
    'like_intensity': 0.8,
    'share_intensity': 0.6,
    'comment_intensity': 0.4,
    'interaction_types': ['like', 'share', 'comment']
}

user_images = []  # List of image paths or URLs
user_connections = []  # List of user IDs

coordinate, confidence = stap.generate_semantic_coordinate(
    user_id="user123",
    user_corpus=user_corpus,
    engagement_patterns=engagement_patterns,
    user_images=user_images,
    user_connections=user_connections
)

print(f"Semantic coordinate shape: {coordinate.shape}")
print(f"Confidence score: {confidence:.3f}")
```

### Finding Bridges

```python
from word2world import EnhancedBridge2World

# Initialize bridge recommendation system
bridge_system = EnhancedBridge2World(
    stap_layer=stap,
    siamese_network=None,  # Will be loaded from checkpoint
    k_neighbors=10,
    k_bridges=5,
    k_horizons=3
)

# Get bridge recommendations for a user
recommendations = bridge_system.recommend_bridges(
    user_id="user123",
    user_coordinate=coordinate,
    all_users=user_database,  # Your user database
    content_pool=content_database  # Your content database
)

print(f"Neighbors: {recommendations['neighbors']}")
print(f"Bridges: {recommendations['bridges']}")
print(f"Horizons: {recommendations['horizons']}")
```

---

## Configuration

### STAPv4Config

```python
config = STAPv4Config(
    # Dimensionality
    target_dim=32,              # Low-dimensional embedding size
    d_text=384,                 # Sentence-BERT dimension
    d_image=512,                # CLIP dimension
    d_network=128,              # Node2Vec dimension
    d_engagement=32,            # Engagement dimension
    
    # Modality weights
    alpha_text=0.54,            # Text weight
    alpha_image=0.18,           # Image weight
    alpha_network=0.18,         # Network weight
    alpha_engagement=0.10,      # Engagement weight
    
    # Covariance estimation
    covariance_method='mcd',    # 'mcd', 'shrinkage', 'diagonal'
    mcd_support_fraction=0.75,  # MCD support fraction
    low_rank_k=None,            # Auto: ⌈√d⌉
    
    # Hyperbolic parameters
    hyperbolic_curvature=-1.0,  # Negative curvature
    a_param=1.577,              # Heavy-tailed distribution
    b_param=0.895,              # Heavy-tailed distribution
    
    # Optimization
    n_epochs=200,               # Training epochs
    learning_rate=1.0,          # Initial learning rate
    batch_size=256,             # Batch size
    
    # Regularization
    alpha_contrast=0.5,         # Contrastive loss weight
    beta_boundary=0.3,          # Boundary penalty weight
    gamma_manifold=0.1,         # Manifold regularization weight
    
    # Performance
    use_gpu=True,               # GPU acceleration
    num_workers=4,              # Data loading workers
    cache_embeddings=True       # Cache computed embeddings
)
```

### CovarianceConfig

```python
from word2world import CovarianceConfig

cov_config = CovarianceConfig(
    method='mcd',               # 'classical', 'mcd', 'shrinkage', 'diagonal'
    mcd_support_fraction=0.75,  # MCD support fraction
    shrinkage_target='identity',# 'identity', 'diagonal', 'constant_correlation'
    low_rank_k=None,            # Auto or specify rank
    use_diagonal_approx=False,  # Use diagonal approximation
    regularization=1e-6         # Regularization for numerical stability
)
```

---

## API Reference

### STAPv4PreprocessingLayer

**Main Methods:**

```python
# Generate semantic coordinate
coordinate, confidence = stap.generate_semantic_coordinate(
    user_id: str,
    user_corpus: List[str],
    engagement_patterns: Dict,
    user_images: List[str] = [],
    user_connections: List[str] = []
) -> Tuple[np.ndarray, float]

# Find nearest neighbors
neighbor_indices, distances = stap.find_nearest_neighbors(
    coordinate: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]

# Update coordinate incrementally
new_coordinate = stap.update_coordinate(
    user_id: str,
    old_coordinate: np.ndarray,
    new_content: List[str],
    alpha: float = 0.1
) -> np.ndarray

# Get coordinate confidence
confidence = stap.get_coordinate_confidence(
    user_id: str,
    coordinate: np.ndarray
) -> float
```

### EnhancedBridge2World

**Main Methods:**

```python
# Recommend bridges
recommendations = bridge_system.recommend_bridges(
    user_id: str,
    user_coordinate: np.ndarray,
    all_users: Dict,
    content_pool: List,
    context: Dict = None
) -> Dict[str, List]

# Compute bridge score
score = bridge_system.compute_bridge_score(
    user_i: np.ndarray,
    user_j: np.ndarray,
    context: Dict
) -> float

# Visualize semantic space
bridge_system.visualize_semantic_space(
    coordinates: np.ndarray,
    labels: List[str],
    bridges: List[Tuple],
    save_path: str = "semantic_space.html"
)
```

### RobustCovarianceEstimator

**Main Methods:**

```python
# Fit covariance matrix
estimator.fit(X: np.ndarray) -> RobustCovarianceEstimator

# Compute Mahalanobis distance
distance = estimator.mahalanobis_distance(
    x_i: np.ndarray,
    x_j: np.ndarray
) -> float

# Get covariance matrix
cov_matrix = estimator.get_covariance() -> np.ndarray

# Get precision matrix
precision = estimator.get_precision() -> np.ndarray
```

---

## Performance

### Computational Complexity

| Operation | Complexity | Time (1000 users) |
|-----------|------------|-------------------|
| High-dim probabilities | O(nk) | ~2.3s |
| STAP optimization | O(E×n×k×d) | ~45s (200 epochs) |
| HNSW construction | O(n log n) | ~0.8s |
| Coordinate generation | O(log n) | ~3ms |
| Nearest neighbor query | O(log n) | ~1ms |
| Bridge scoring (Siamese) | O(d) | ~0.5ms |

### Quality Metrics

- **Neighbor preservation:** 72.3% (±3.1%)
- **Bridge identification accuracy:** 68.4%
- **Silhouette score:** 0.61
- **Bridge acceptance rate:** 43.7% (vs. 28.3% baseline)
- **Clustering purity (NMI):** 0.73

### Scalability

- **100 users:** ~5s total processing
- **1,000 users:** ~48s total processing
- **10,000 users:** ~8min total processing
- **1M+ users:** Supported via HNSW indexing + diagonal approximation

### Memory Usage

- **Full covariance (d=1056):** 8.9 MB
- **Low-rank approximation (k=32):** 0.27 MB (97% reduction)
- **Diagonal approximation:** 0.008 MB (99.9% reduction)

---

## Deployment

### Local Development

```bash
# Clone repository
git clone https://github.com/word2world/stap.git
cd stap

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_token"

# Run tests
pytest tests/

# Start development server
python app.py
```

### Google Colab

1. Upload `word2world_v4_github.py` to Google Colab
2. Add Hugging Face token to Colab Secrets (🔑 icon)
3. Run all cells

### Production Deployment

```bash
# Install with production dependencies
pip install -r requirements-prod.txt

# Configure environment
export HF_TOKEN="your_token"
export STAP_CONFIG="production"
export GPU_ENABLED="true"

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## Research

### Publications

- **Technical Paper:** "Multi-Modal Hyperbolic Embeddings for Peace-Building Social Networks" (2025)
- **Pilot Study:** "Word2World MIIS Deployment: Evaluation Report" (forthcoming)

### Key References

- Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. *NeurIPS*.
- Chami, I., et al. (2019). Hyperbolic Graph Convolutional Neural Networks. *NeurIPS*.
- Chami, I., et al. (2020). Machine learning on hyperbolic spaces: Concepts and applications. *NeurIPS*.
- Bu, Y., et al. (2025). Boundary-aware optimization for hyperbolic neural networks. *ICLR*.
- Ganea, O.-E., et al. (2018). Hyperbolic Neural Networks. *NeurIPS*.
- Davidson, T. R., et al. (2018). Hyperspherical Variational Auto-Encoders. *UAI*.
- Sala, F., et al. (2018). Representation tradeoffs for hyperbolic embeddings. *ICML*.

---

## Contributing

We welcome contributions from developers, researchers, and inspired peace-builders!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Areas

- Algorithm improvements
- Frontend development
- Testing and benchmarks
- Documentation
- Localization
- Research validation

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Project Website:** [Coming Soon]
- **Email:** Valcin, Vieux - Technical Lead, Full-Stack [vvalcin@middlebury.edu]

---

## Acknowledgments

- **Kathryn W. Davis Projects for Peace Committee** - Funding opportunity through the Davis Projects for Peace program
- **Middlebury Institute of International Studies** - Institutional support and resources
- **Sentence Transformers** - Pre-trained models for semantic embeddings
- **OpenAI CLIP** - Vision-language models
- **HNSW Library** - Efficient nearest neighbor search
- **Open Source Community** - Countless tools and libraries that made this possible

---

## References

[1] Human Rights Council. (2018). Report of the independent international fact-finding mission on Myanmar. https://www.ohchr.org/sites/default/files/Documents/HRBodies/HRCouncil/FFM-Myanmar/A_HRC_39_64.pdf

[2] Islam, M. S., Sarkar, T., Khan, S. H., Mostofa Kamal, A. H., Hasan, S. M. M., Kabir, A., Yeasmin, D., Islam, M. A., Amin Chowdhury, K. I., Anwar, K. S., Chughtai, A. A., & Seale, H. (2020). COVID-19-Related Infodemic and Its Impact on Public Health: A Global Social Media Analysis. The American journal of tropical medicine and hygiene, 103(4), 1621–1629. https://doi.org/10.4269/ajtmh.20-0812

---

<div align="center">

**⭐ Star this repository if you believe social media can be a force for peace ⭐**

Made with ❤️ for a more peaceful world

</div>

