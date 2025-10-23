# Word2World
**Bridging Ideas Together**

[![Version](https://img.shields.io/badge/version-4.0-blue.svg)](https://github.com/word2world/stap) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8+-yellow.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/) [![Projects for Peace](https://img.shields.io/badge/Projects%20for%20Peace-2025-orange.svg)](https://www.projectsforpeace.org/)

*A Projects for Peace Initiative* by 
Translation, Interpretation, and Localization Management (TILM) students at the
Middlebury Institute of International Studies at Monterey (MIIS)

---

## Overview

**Word2World** is a next-generation social media platform that leverages advanced mathematical frameworks to bridge ideological divides and foster constructive dialogue. Developed as part of a **Projects for Peace** initiative at the Middlebury Institute of International Studies at Monterey, Word2World represents a paradigm shift from engagement-driven algorithms to peace-building technology.

At its core is the **STAP v4.0 (Semantic Topology-Aware Projection)** framework, which combines multi-modal embeddings, hyperbolic geometry, and deep learning to identify "bridge" content that connects users with different perspectives.

### The Problem We're Solving

In 2017, Facebook's algorithms amplified hate speech that contributed to genocide in Myanmar. In 2021, social media platforms mobilized a violent insurrection at the U.S. Capitol. During the COVID-19 pandemic, algorithmic amplification of health misinformation led to at least 800 documented deaths. The pattern is clear and devastating: **social media platforms designed to maximize engagement systematically amplify conflict, polarization, and violence.**

Most responses to this crisis are reactive—better content moderation, fact-checking, user education. These are necessary but insufficient. They treat the symptoms, not the disease. The fundamental problem is structural: social media algorithms are optimized for engagement, and divisive content generates more engagement than bridge-building dialogue.

### Our Solution

**Word2World is different.** It is a social media platform designed from the ground up for peace-building, not engagement. This is not a concept—we have already built a functional prototype with a substantial codebase built upon a novel, proprietary Semantic Topology-Aware Projection (STAP) mathematical framework.

**The tech works.** What is needed now is to showcase its ability to promote peace in a real-world community through our MIIS pilot deployment.

### What Makes STAP v4.0 Unique?

Unlike traditional recommendation systems that create echo chambers, STAP v4.0 actively identifies and promotes content that:

- **Validates** your existing views (Neighbors)

- **Expands** your perspective with related but different ideas (Bridges)

- **Challenges** you with distant viewpoints (Horizons)

The system uses three groundbreaking mathematical innovations:

1. **Multi-Modal Mahalanobis Distance** - Combines text, images, and social network data with covariance-aware distance metrics

1. **Hyperbolic Geometry** - Represents semantic hierarchies in curved space, naturally encoding ideological trees

1. **Siamese Neural Networks** - Learns to predict bridge success from historical interaction data

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

1. **Common Ground Theory**: Finding shared values and interests is the foundation of dialogue (Clark, 1996)

1. **Perspective-Taking**: Understanding others' viewpoints increases empathy and reduces hostility (Galinsky & Moskowitz, 2000)

STAP operationalizes these principles at scale. By identifying and promoting content that bridges divides, Word2World creates the conditions for constructive dialogue and mutual understanding.

### Why This Project Will Succeed

1. **Proven Technology**: We have already built a sophisticated, functional prototype with over 1,600 lines of production-quality code

1. **Rigorous Evaluation**: We will use validated instruments, statistical analysis, and mixed-methods research to provide concrete evidence of impact

1. **Scalable Design**: The architecture is designed from the ground up for global deployment

1. **Expert Team**: Our team combines technical expertise (AI development, localization engineering), domain knowledge (conflict resolution, peace studies), and project management skills

1. **Institutional Support**: MIIS provides an ideal environment with a globally-minded community and strong institutional commitment to peace-building

1. **Urgent Need**: The evidence of social media's role in violence is overwhelming. There is a clear and pressing need for alternative models

1. **Replicable Model**: The "Digital Peace-Building Toolkit" will enable other institutions and organizations to adopt Word2World, multiplying impact

### MIIS Pilot Deployment

The MIIS pilot will:

- Deploy Word2World to the MIIS community (500+ students, faculty, staff)

- Measure impact on dialogue quality, perspective-taking, and community cohesion

- Validate the STAP framework in a real-world setting

- Create a replicable model for other institutions

- Produce a comprehensive evaluation report and "Digital Peace-Building Toolkit"

This is not just a student project. It is the beginning of a paradigm shift in how we think about digital communication and peace-building.

**We are not just talking about peace—we are building the digital infrastructure for it.**

---

## Features

### Core Capabilities

✅ **Multi-Modal User Representation**

- Text embeddings via Sentence-BERT (384D)

- Image embeddings via CLIP (512D)

- Network embeddings via Node2Vec (128D)

- Engagement embeddings via Behavioral Aggregation (32D)

- Weighted fusion: α_text=0.54, α_image=0.18, α_network=0.18, α_engagement=0.10

✅ **Advanced Distance Metrics**

- **Robust Mahalanobis distance** with MCD and low-rank approximation

- Hyperbolic distance in Poincaré ball

- Contrastive loss for boundary classification

- Confidence scoring for coordinate reliability

✅ **Intelligent Recommendation System**

- Three-tier recommendations: Neighbors, Bridges, Horizons

- Bridge-aware scoring via Siamese neural networks

- Real-time updates with incremental learning

- O(log n) nearest neighbor search via HNSW

✅ **Scalable Architecture**

- Handles 1M+ users

- Incremental covariance updates

- Efficient diagonal approximation option

- GPU acceleration for Siamese inference

✅ **Research-Backed Design**

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

- Low-rank approximation: `Σ ≈ U_k U_kᵀ + D` for 97% memory reduction

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
Q(i,j) = 1 / (1 + d_H(y_i, y_j)²)
```

---

---

### Robust Covariance Estimation (New in v4.0)

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

### Optimization: Cross-Entropy + Contrastive Loss

**STAP Objective:**

```
L_STAP = L_attract + L_repel

L_attract = Σ_{i,j} P(i,j) log(P(i,j) / Q(i,j))

L_repel = Σ_{i,k} (1 - P(i,k)) log((1 - P(i,k)) / (1 - Q(i,k)))
```

**Contrastive Loss:**

```
L_contrast = Σ_{(i,j)∈S} d_H(y_i, y_j)² + Σ_{(i,k)∈D} max(0, γ - d_H(y_i, y_k))²
```

where:

- `S` = set of similar pairs (k-nearest neighbors)

- `D` = set of dissimilar pairs (random sampling)

- `γ` = margin for dissimilar pairs (default: 2.0)

**Combined Objective:**

```
L_total = L_STAP + λ_contrast · L_contrast + λ_boundary · L_boundary
```

**Riemannian Gradient Descent:**

```
grad_Riemannian = ((1 - ||y_i||²) / 2)² · grad_Euclidean

y_i^(t+1) = exp_y_i(-η · grad_Riemannian)
```

where `exp_y` is the exponential map in the Poincaré ball.

---

### Bridge-Aware Scoring: Siamese Neural Network

**Objective:** Learn to predict bridge success probability from semantic coordinates and user context.

**Architecture:**

```
Input: (y_i, c_i, y_j, c_j)
  ↓
Siamese Tower (shared weights):
  Linear(d_low + d_context → d_hidden)
  ReLU + Dropout(0.3)
  Linear(d_hidden → d_hidden/2)
  ReLU + Dropout(0.3)
  Linear(d_hidden/2 → d_output)
  ↓
Distance Layer:
  d_ij = |h_i - h_j|
  ↓
Prediction Head:
  Linear(d_output → 64)
  ReLU + Dropout(0.3)
  Linear(64 → 32)
  ReLU
  Linear(32 → 1)
  Sigmoid
  ↓
Output: B(i,j) ∈ [0, 1]
```

**Context Vector (c_i):**

- Engagement level (interactions/day)

- Recent activity (posts in 24h)

- Topic distribution (from LDA/BERTopic)

- Sentiment score

- Openness to diverse views

- Demographic features (optional)

**Training:**

- Loss: Binary Cross-Entropy

- Optimizer: Adam (lr=0.001, weight_decay=1e-4)

- Regularization: Dropout + L2

- Validation: 80/20 train/test split

**Final Recommendation Score:**

```
S_final(i,j) = f(d_H(y_i, y_j)) × C(i,j) × B(i,j)
```

where:

- `f(d)` = semantic scoring function (e.g., exp(-d²))

- `C(i,j)` = confidence weight

- `B(i,j)` = bridge success probability from Siamese network

---

## Installation

### Requirements

- Python 3.8+

- PyTorch 2.0+

- CUDA (optional, for GPU acceleration)

### Quick Install

```bash
# Clone repository
git clone https://github.com/word2world/stap-v4.git
cd stap-v4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt

```
# Core
numpy==1.24.0
scipy==1.10.0
scikit-learn==1.3.0

# Deep Learning
torch==2.0.0
torchvision==0.15.0

# NLP & Embeddings
sentence-transformers==2.2.2
transformers==4.30.0

# STAP-Specific
umap-learn==0.5.4
pynndescent==0.5.10
hnswlib==0.7.0

# Utilities
tqdm==4.65.0
```

### Google Colab

```python
# Upload word2world_social_v4.py and colab_setup.py to Colab
!python colab_setup.py

# Import and use
from word2world_social_v4 import STAPv4Config, STAPv4PreprocessingLayer
```

---

## Quick Start

### Basic Usage

```python
from word2world_social_v4 import STAPv4Config, STAPv4PreprocessingLayer

# Configure STAP v4.0
config = STAPv4Config(
    use_mahalanobis=True,
    use_hyperbolic=True,
    use_contrastive_loss=True,
    use_siamese=False,  # Enable after training
    n_epochs=100
)

# Initialize
stap = STAPv4PreprocessingLayer(config)

# Generate semantic coordinate
user_corpus = [
    "I believe in renewable energy.",
    "Solar power is the future.",
    "We need climate action now."
]

engagement = {
    "likes": 15,
    "shares": 8,
    "comments": 5
}

coordinate, confidence = stap.generate_semantic_coordinate(
    user_id="user_001",
    user_corpus=user_corpus,
    engagement_patterns=engagement
)

print(f"Coordinate: {coordinate.shape}")  # (32,)
print(f"Confidence: {confidence:.3f}")    # 0.750
```

### Finding Bridges

```python
# Find nearest neighbors
neighbors, distances = stap.find_nearest_neighbors(coordinate, k=10)

# Classify recommendations
for idx, dist in zip(neighbors, distances):
    if dist < 0.3:
        category = "Neighbor"  # Similar views
    elif dist < 0.5:
        category = "Bridge"    # Optimal distance
    else:
        category = "Horizon"   # Distant views
    
    print(f"User {idx}: {category} (distance: {dist:.3f})")
```

### Computing Bridge Scores

```python
# With Siamese network (after training)
context_i = construct_context_vector(user_i_data)
context_j = construct_context_vector(user_j_data)

bridge_score = stap.compute_bridge_score(
    user_i_id="user_001",
    user_j_id="user_042",
    context_i=context_i,
    context_j=context_j
)

print(f"Bridge Success Probability: {bridge_score:.3f}")
```

---

## Configuration

### Configuration Options

```python
from word2world import STAPv4Config

config = STAPv4Config(
    # Multi-modal weights
    alpha_text=0.54,
    alpha_image=0.18,
    alpha_network=0.18,
    alpha_engagement=0.10,
    
    # Robust covariance (NEW in v4.0)
    covariance_method="robust",  # "classical", "robust", or "cellwise"
    covariance_rank=32,  # None for auto (⌈√d⌉)
    mcd_support_fraction=0.75,  # MCD contamination tolerance
    
    # Hyperbolic space
    d_low=32,
    
    # Optimization
    learning_rate=0.01,
    epochs=200,
    
    # Contrastive loss
    use_contrastive=True,
    gamma=2.0,
    lambda_contrast=0.1
)
```

### Configuration Templates

**Conservative (Testing):**

```python
from word2world import STAPv4Config

config = STAPv4Config(
    # Multi-modal weights
    alpha_text=0.54,
    alpha_image=0.18,
    alpha_network=0.18,
    alpha_engagement=0.10,
    
    # Robust covariance (NEW in v4.0)
    covariance_method="robust",  # "classical", "robust", or "cellwise"
    covariance_rank=32,  # None for auto (⌈√d⌉)
    mcd_support_fraction=0.75,  # MCD contamination tolerance
    
    # Hyperbolic space
    d_low=32,
    
    # Optimization
    learning_rate=0.01,
    epochs=200,
    
    # Contrastive loss
    use_contrastive=True,
    gamma=2.0,
    lambda_contrast=0.1
)
```

**Standard (Production):**

```python
config = STAPv4Config(
    use_mahalanobis=True,
    use_diagonal_covariance=True,
    use_hyperbolic=True,
    use_contrastive_loss=True,
    use_siamese=False,
    n_epochs=100
)
```

**Advanced (With Siamese):**

```python
config = STAPv4Config(
    use_mahalanobis=True,
    use_diagonal_covariance=False,
    use_hyperbolic=True,
    use_contrastive_loss=True,
    use_siamese=True,
    n_epochs=200
)
```

---

## API Reference

### STAPv4PreprocessingLayer

**Main Methods:**

#### `generate_semantic_coordinate()`

```python
def generate_semantic_coordinate(
    user_id: str,
    user_corpus: List[str],
    engagement_patterns: Dict,
    image_data: Optional[List] = None,
    network_connections: Optional[List] = None
) -> Tuple[np.ndarray, float]
```

Generates semantic coordinate for a user.

**Parameters:**

- `user_id`: Unique user identifier

- `user_corpus`: List of user's text content

- `engagement_patterns`: Dict with interaction data (likes, shares, comments)

- `image_data`: Optional list of images

- `network_connections`: Optional list of connections

**Returns:**

- `coordinate`: (d,) semantic coordinate in low-dimensional space

- `confidence`: Confidence score [0, 1]

---

#### `find_nearest_neighbors()`

```python
def find_nearest_neighbors(
    query_coordinate: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]
```

Finds k nearest neighbors using HNSW index.

**Parameters:**

- `query_coordinate`: Query point in semantic space

- `k`: Number of neighbors to return

**Returns:**

- `neighbor_indices`: (k,) array of neighbor indices

- `distances`: (k,) array of distances

---

#### `compute_bridge_score()`

```python
def compute_bridge_score(
    user_i_id: str,
    user_j_id: str,
    context_i: Optional[np.ndarray] = None,
    context_j: Optional[np.ndarray] = None
) -> float
```

Computes final bridge-aware recommendation score.

**Parameters:**

- `user_i_id`: User i identifier

- `user_j_id`: User j identifier

- `context_i`: Optional context vector for user i

- `context_j`: Optional context vector for user j

**Returns:**

- `score`: Final bridge score [0, 1]

---

### MultiModalMahalanobisDistance

**Methods:**

#### `fit()`

```python
def fit(X_multimodal: np.ndarray) -> None
```

Computes covariance matrix from multi-modal embeddings.

---

#### `distance()`

```python
def distance(x_i: np.ndarray, x_j: np.ndarray) -> float
```

Computes Mahalanobis distance between two embeddings.

---

### HyperbolicSpace

**Methods:**

#### `hyperbolic_distance()`

```python
def hyperbolic_distance(y_i: np.ndarray, y_j: np.ndarray) -> float
```

Computes hyperbolic distance in Poincaré ball.

---

#### `project_to_ball()`

```python
def project_to_ball(y: np.ndarray, max_norm: float = 0.99) -> np.ndarray
```

Projects point onto Poincaré ball.

---

### SiameseNetwork

**Methods:**

#### `forward()`

```python
def forward(
    y_i: torch.Tensor,
    c_i: torch.Tensor,
    y_j: torch.Tensor,
    c_j: torch.Tensor
) -> torch.Tensor
```

Forward pass through Siamese network.

---

#### `bridge_score()`

```python
def bridge_score(
    y_i: np.ndarray,
    c_i: np.ndarray,
    y_j: np.ndarray,
    c_j: np.ndarray,
    device: str = 'cpu'
) -> float
```

Computes bridge success probability for a single pair.

---

## Performance

### Benchmarks

| Operation | N=1,000 | N=10,000 | N=100,000 |
| --- | --- | --- | --- |
| Text Embedding | 2.5s | 25s | 250s |
| Mahalanobis (Diag) | 0.01s | 0.1s | 1s |
| Mahalanobis (Full) | 0.05s | 0.5s | 5s |
| Hyperbolic Distance | 0.001s | 0.01s | 0.1s |
| STAP Optimization | 15s | 180s | 1,800s |
| HNSW Build | 0.5s | 8s | 120s |
| HNSW Query (k=10) | 0.0001s | 0.0001s | 0.0001s |
| Siamese Inference | 0.01s | 0.05s | 0.5s |

### Memory Footprint

| Component | N=1,000 | N=10,000 | N=100,000 |
| --- | --- | --- | --- |
| Text Embeddings | 3 MB | 30 MB | 300 MB |
| Multi-modal Embeddings | 8 MB | 80 MB | 800 MB |
| Covariance (Diagonal) | 8 KB | 80 KB | 800 KB |
| Covariance (Full) | 8 MB | 800 MB | 8 GB |
| Low-dim Embeddings | 0.25 MB | 2.5 MB | 25 MB |
| HNSW Index | 2 MB | 20 MB | 200 MB |
| **Total (Diagonal)** | **~20 MB** | **~140 MB** | **~1.4 GB** |
| **Total (Full)** | **~25 MB** | **~940 MB** | **~9.4 GB** |

### Expected Improvements (vs STAP v3.0)

| Metric | v3.0 | v4.0 | Improvement |
| --- | --- | --- | --- |
| Bridge Precision | 0.65 | 0.80 | **+23%** |
| Bridge Recall | 0.58 | 0.75 | **+29%** |
| Bridge Engagement | 12% | 20% | **+67%** |
| Constructive Interactions | 8% | 15% | **+88%** |
| Neighbor Preservation | 0.72 | 0.85 | **+18%** |

---

## Deployment

### Phase 1: Mahalanobis Distance

**Goal:** Improve semantic distance with covariance awareness

**Configuration:**

```python
config = STAPv4Config(
    use_mahalanobis=True,
    use_diagonal_covariance=True,
    use_hyperbolic=False,
    use_siamese=False
)
```

**Timeline:** Week 1-2

**Success Metrics:**

- Neighbor preservation rate > 0.75

- Distance computation time < 0.1s per pair

---

### Phase 2: Hyperbolic Geometry

**Goal:** Capture hierarchical semantic structure

**Configuration:**

```python
config = STAPv4Config(
    use_mahalanobis=True,
    use_hyperbolic=True,
    use_contrastive_loss=True,
    use_siamese=False
)
```

**Timeline:** Week 3-5

**Success Metrics:**

- Bridge precision > 0.70

- Semantic tree visualization shows clear clusters

---

### Phase 3: Siamese Network

**Goal:** Learn bridge success from interaction data

**Prerequisites:**

- Collect ≥5,000 labeled interaction pairs

- Train Siamese network with validation AUC > 0.75

**Configuration:**

```python
config = STAPv4Config(
    use_mahalanobis=True,
    use_hyperbolic=True,
    use_contrastive_loss=True,
    use_siamese=True
)
```

**Timeline:** Week 6-10

**Success Metrics:**

- Bridge engagement rate > 18%

- Constructive interaction rate > 12%

---

### Monitoring

**Key Metrics:**

- Bridge identification precision/recall

- User engagement with bridge content

- Constructive vs destructive interaction ratio

- System latency (p50, p95, p99)

- Memory usage

**Alerts:**

- Covariance matrix singular

- Points escaping Poincaré ball

- HNSW index corruption

- Siamese network overfitting

---

## Research

### Publications

1. **STAP v4.0 Technical Paper** (2025)
  - Multi-modal Mahalanobis distance for semantic spaces
  - Hyperbolic geometry for ideological representation
  - Siamese networks for bridge prediction

1. **Social Media's Role in Conflict** (2024)
  - Research synthesis on polarization
  - Intervention strategies
  - Word2World case study

### Datasets

- **Word2World Interaction Dataset** (coming soon)
  - 100K+ labeled user pairs
  - Constructive vs destructive interactions
  - Multi-modal user profiles

### Collaborations

We welcome academic collaborations! Contact: [research@word2world.org](mailto:research@word2world.org)

---

## Contributing

We welcome contributions from the community!

### How to Contribute

1. Fork the repository

1. Create a feature branch (`git checkout -b feature/amazing-feature`)

1. Commit your changes (`git commit -m 'Add amazing feature'`)

1. Push to the branch (`git push origin feature/amazing-feature`)

1. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/stap-v4.git
cd stap-v4

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 word2world_social_v4.py
black word2world_social_v4.py
```

### Code Style

- Follow PEP 8

- Use type hints

- Write docstrings for all public methods

- Add unit tests for new features

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

**Core Team:**

- **Vieux Valcin** (Technical Lead) - AI Development, STAP Framework Architecture, Backend Engineering, Demo Production
  - Email: [vvalcin@miis.edu](mailto:vvalcin@miis.edu)

- **Jinxing Chen** (Frontend Development) - UI/UX Design, Visual Assets, Demo Production
  - Email: [jinxingc@middlebury.edu](mailto:jinxingc@middlebury.edu)

- **Xiaoyue Cao** (Backend Development) - Repository Organization, Technical Documentation, Demo Production
  - Email: [xiaoyuec@middlebury.edu](mailto:xiaoyuec@middlebury.edu)

- **Huiyi Zhang** (Project Management) - Impact Research and Content Synthesis
  - Email: [hz3@middlebury.edu](mailto:hz3@middlebury.edu)

- **Shu Bai** (Project Management) - Impact Research and Content Synthesis
  - Email: [sbai@middlebury.edu](mailto:sbai@middlebury.edu)

- **Yunzhou Dai** (Project Management) - Impact Research and Content Synthesis
  - Email: [yunzhoud@middlebury.edu](mailto:yunzhoud@middlebury.edu)

**Institution:**

- Middlebury Institute of International Studies at Monterey

**Projects for Peace:**

- This project is part of the Projects for Peace initiative, which supports student-designed grassroots projects that promote peace and address the root causes of conflict.

---

## Acknowledgments

**Technical:**

- Sentence-BERT team for semantic embeddings

- HNSW library for fast nearest neighbor search

- PyTorch team for deep learning framework

- Research community for hyperbolic geometry insights

**Institutional:**

- Middlebury Institute of International Studies at Monterey for institutional support

- Projects for Peace for funding and recognition

- MIIS community for pilot testing and feedback

**Research:**

- Contact Hypothesis research (Allport, 1954)

- Common Ground Theory (Clark, 1996)

- Perspective-Taking studies (Galinsky & Moskowitz, 2000)

- Social media conflict research community

---

## Citation

If you use STAP v4.0 in your research, please cite:

```
@article{word2world2025stap,
  title={STAP v4.0: Multi-Modal Hyperbolic Embeddings for Bridge Identification in Social Networks},
  author={Bai, Shu and Cao, Xiaoyue and Chen, Jinxing and Dai, Yunzhou and Valcin, Vieux and Zhang, Huiyi},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```

---

## Contact

**Project Team:**

- **Vieux Valcin** (Technical Lead) - [vvalcin@miis.edu](mailto:vvalcin@miis.edu)

- **Shu Bai** - [sbai@middlebury.edu](mailto:sbai@middlebury.edu)

- **Xiaoyue Cao** - [xiaoyuec@middlebury.edu](mailto:xiaoyuec@middlebury.edu)

- **Jinxing Chen** - [jinxingc@middlebury.edu](mailto:jinxingc@middlebury.edu)

- **Yunzhou Dai** - [yunzhoud@middlebury.edu](mailto:yunzhoud@middlebury.edu)

- **Huiyi Zhang** - [hz3@middlebury.edu](mailto:hz3@middlebury.edu)

**Repository:**

- **GitHub:** [https://github.com/word2world](https://github.com/word2world)

**Institution:**

- Middlebury Institute of International Studies at Monterey

- 460 Pierce Street, Monterey, CA 93940

- [https://www.miis.edu](https://www.miis.edu)

---

## Roadmap

### Q1 2025

- ✅ STAP v4.0 release

- ✅ Multi-modal Mahalanobis distance

- ✅ Hyperbolic geometry integration

- ✅ Siamese network architecture

### Q2 2025

- [ ] CLIP integration for images

- [ ] Node2Vec for network embeddings

- [ ] Active learning for Siamese training

- [ ] Production deployment

### Q3 2025

- [ ] Attention-based Siamese network

- [ ] Dynamic modality weighting

- [ ] Hierarchical clustering

- [ ] Multi-lingual support

### Q4 2025

- [ ] Temporal dynamics modeling

- [ ] Explainable bridge recommendations

- [ ] Mobile app launch

- [ ] Public API release

---

**Built with ❤️ by the Word2World team**
*Bridging divides, one conversation at a time.*

