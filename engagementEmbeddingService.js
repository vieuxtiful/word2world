/**
 * engagementEmbeddingService.js
 * 
 * Service for computing engagement embeddings (x_i^engagement ∈ R^32)
 * from user engagement events.
 * 
 * Processes: views, comments, hashtags, likes, saves
 * Outputs: 32-dimensional engagement embedding vector
 * 
 * Usage:
 *   const { computeEngagementEmbedding } = require('./services/engagementEmbeddingService');
 *   const embedding = await computeEngagementEmbedding(userId);
 */

const { Pool } = require('pg');
const natural = require('natural'); // For text processing
const math = require('mathjs');

const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'word2world',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD,
});

// Subweights for engagement types (β values)
const SUBWEIGHTS = {
  views: 0.35,
  comments: 0.30,
  hashtags: 0.20,
  likes: 0.10,
  saves: 0.05,
};

// Dimensional allocation
const DIMENSIONS = {
  views: { topic: 5, engagement: 4, temporal: 2, total: 11 },
  comments: { topic: 4, engagement: 4, temporal: 2, total: 10 },
  hashtags: { topic: 3, engagement: 2, temporal: 1, total: 6 },
  likes: { topic: 1, engagement: 1, temporal: 1, total: 3 },
  saves: { topic: 1, engagement: 1, temporal: 0, total: 2 },
};

/**
 * Main function: Compute engagement embedding for a user
 */
async function computeEngagementEmbedding(userId, timeWindow = '90 days') {
  try {
    // Fetch all engagement events for user
    const events = await fetchEngagementEvents(userId, timeWindow);
    
    if (events.length === 0) {
      // Return zero vector if no engagement
      return Array(32).fill(0);
    }

    // Compute features for each engagement type
    const viewsFeatures = await computeViewsFeatures(events.views);
    const commentsFeatures = await computeCommentsFeatures(events.comments);
    const hashtagsFeatures = await computeHashtagsFeatures(events.hashtags);
    const likesFeatures = await computeLikesFeatures(events.likes);
    const savesFeatures = await computeSavesFeatures(events.saves);

    // Apply subweights and concatenate
    const embedding = [
      ...viewsFeatures.map(v => v * SUBWEIGHTS.views),
      ...commentsFeatures.map(v => v * SUBWEIGHTS.comments),
      ...hashtagsFeatures.map(v => v * SUBWEIGHTS.hashtags),
      ...likesFeatures.map(v => v * SUBWEIGHTS.likes),
      ...savesFeatures.map(v => v * SUBWEIGHTS.saves),
    ];

    // Normalize to unit length (optional, depends on STAP implementation)
    const normalized = normalizeVector(embedding);

    return normalized;
  } catch (error) {
    console.error('Error computing engagement embedding:', error);
    throw error;
  }
}

/**
 * Fetch engagement events grouped by type
 */
async function fetchEngagementEvents(userId, timeWindow) {
  const result = await pool.query(
    `SELECT event_type, event_data, timestamp
     FROM engagement_events
     WHERE user_id = $1 
       AND timestamp > NOW() - INTERVAL '${timeWindow}'
     ORDER BY timestamp DESC`,
    [userId]
  );

  // Group by event type
  const grouped = {
    views: [],
    comments: [],
    hashtags: [],
    likes: [],
    saves: [],
  };

  result.rows.forEach(row => {
    if (grouped[row.event_type]) {
      grouped[row.event_type].push({
        data: row.event_data,
        timestamp: row.timestamp,
      });
    }
  });

  return grouped;
}

/**
 * Compute Views features (11 dimensions)
 * - Topic distribution (5 dims)
 * - Engagement metrics (4 dims): duration, frequency, diversity, scroll_depth
 * - Temporal patterns (2 dims): recency, consistency
 */
async function computeViewsFeatures(viewEvents) {
  if (viewEvents.length === 0) {
    return Array(DIMENSIONS.views.total).fill(0);
  }

  // Fetch post content for viewed posts
  const postIds = viewEvents.map(e => e.data.post_id);
  const posts = await fetchPostsByIds(postIds);

  // Topic distribution (simplified: use TF-IDF of top words)
  const topicDist = computeTopicDistribution(posts, DIMENSIONS.views.topic);

  // Engagement metrics
  const avgDuration = viewEvents.reduce((sum, e) => sum + (e.data.duration_ms || 0), 0) / viewEvents.length / 1000; // seconds
  const frequency = viewEvents.length / 90; // views per day (assuming 90-day window)
  const diversity = computeTopicDiversity(posts);
  const avgScrollDepth = viewEvents.reduce((sum, e) => sum + (e.data.scroll_depth || 0), 0) / viewEvents.length;

  // Temporal patterns
  const recency = computeRecencyScore(viewEvents);
  const consistency = computeConsistencyScore(viewEvents);

  return [
    ...topicDist,                                    // 5 dims
    normalize(avgDuration, 0, 300),                  // 1 dim (0-5 min)
    normalize(frequency, 0, 50),                     // 1 dim (0-50 views/day)
    diversity,                                       // 1 dim
    avgScrollDepth,                                  // 1 dim
    recency,                                         // 1 dim
    consistency,                                     // 1 dim
  ];
}

/**
 * Compute Comments features (10 dimensions)
 */
async function computeCommentsFeatures(commentEvents) {
  if (commentEvents.length === 0) {
    return Array(DIMENSIONS.comments.total).fill(0);
  }

  const postIds = commentEvents.map(e => e.data.post_id);
  const posts = await fetchPostsByIds(postIds);

  const topicDist = computeTopicDistribution(posts, DIMENSIONS.comments.topic);

  const frequency = commentEvents.length / 90;
  const avgLength = commentEvents.reduce((sum, e) => sum + (e.data.comment_length || 0), 0) / commentEvents.length;
  const diversity = computeTopicDiversity(posts);
  const avgReplyDepth = commentEvents.reduce((sum, e) => sum + (e.data.reply_depth || 0), 0) / commentEvents.length;

  const recency = computeRecencyScore(commentEvents);
  const consistency = computeConsistencyScore(commentEvents);

  return [
    ...topicDist,                                    // 4 dims
    normalize(frequency, 0, 10),                     // 1 dim
    normalize(avgLength, 0, 500),                    // 1 dim
    diversity,                                       // 1 dim
    normalize(avgReplyDepth, 0, 5),                  // 1 dim
    recency,                                         // 1 dim
    consistency,                                     // 1 dim
  ];
}

/**
 * Compute Hashtags features (6 dimensions)
 */
async function computeHashtagsFeatures(hashtagEvents) {
  if (hashtagEvents.length === 0) {
    return Array(DIMENSIONS.hashtags.total).fill(0);
  }

  // Extract hashtags
  const hashtags = hashtagEvents.map(e => e.data.hashtag);
  
  // Topic distribution via hashtag clustering (simplified: top hashtags)
  const hashtagCounts = {};
  hashtags.forEach(tag => {
    hashtagCounts[tag] = (hashtagCounts[tag] || 0) + 1;
  });
  
  const topHashtags = Object.entries(hashtagCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, DIMENSIONS.hashtags.topic);
  
  const topicDist = topHashtags.map(([tag, count]) => count / hashtags.length);
  while (topicDist.length < DIMENSIONS.hashtags.topic) {
    topicDist.push(0);
  }

  const frequency = hashtagEvents.length / 90;
  const uniqueCount = Object.keys(hashtagCounts).length;
  const diversity = uniqueCount / hashtags.length; // ratio of unique to total

  const recency = computeRecencyScore(hashtagEvents);

  return [
    ...topicDist,                                    // 3 dims
    normalize(frequency, 0, 20),                     // 1 dim
    normalize(diversity, 0, 1),                      // 1 dim
    recency,                                         // 1 dim
  ];
}

/**
 * Compute Likes features (3 dimensions)
 */
async function computeLikesFeatures(likeEvents) {
  if (likeEvents.length === 0) {
    return Array(DIMENSIONS.likes.total).fill(0);
  }

  const postIds = likeEvents.map(e => e.data.post_id);
  const posts = await fetchPostsByIds(postIds);

  // Dominant topic (mode)
  const topics = posts.map(p => extractDominantTopic(p.content));
  const topicMode = mode(topics);

  const frequency = likeEvents.length / 90;
  const recency = computeRecencyScore(likeEvents);

  return [
    topicMode,                                       // 1 dim (encoded as 0-1)
    normalize(frequency, 0, 100),                    // 1 dim
    recency,                                         // 1 dim
  ];
}

/**
 * Compute Saves features (2 dimensions)
 */
async function computeSavesFeatures(saveEvents) {
  if (saveEvents.length === 0) {
    return Array(DIMENSIONS.saves.total).fill(0);
  }

  const postIds = saveEvents.map(e => e.data.post_id);
  const posts = await fetchPostsByIds(postIds);

  const topics = posts.map(p => extractDominantTopic(p.content));
  const topicMode = mode(topics);

  const frequency = saveEvents.length / 90;

  return [
    topicMode,                                       // 1 dim
    normalize(frequency, 0, 10),                     // 1 dim
  ];
}

/**
 * Helper: Fetch posts by IDs
 */
async function fetchPostsByIds(postIds) {
  if (postIds.length === 0) return [];
  
  const result = await pool.query(
    `SELECT id, content FROM posts WHERE id = ANY($1)`,
    [postIds]
  );
  
  return result.rows;
}

/**
 * Helper: Compute topic distribution using TF-IDF (simplified)
 */
function computeTopicDistribution(posts, numTopics) {
  if (posts.length === 0) {
    return Array(numTopics).fill(0);
  }

  // Simple bag-of-words approach (in production, use LDA or Sentence-BERT)
  const TfIdf = natural.TfIdf;
  const tfidf = new TfIdf();

  posts.forEach(post => {
    tfidf.addDocument(post.content);
  });

  // Get top terms
  const topTerms = [];
  tfidf.listTerms(0).slice(0, numTopics).forEach(item => {
    topTerms.push(item.tfidf);
  });

  // Normalize to sum to 1
  const sum = topTerms.reduce((a, b) => a + b, 0) || 1;
  const normalized = topTerms.map(v => v / sum);

  while (normalized.length < numTopics) {
    normalized.push(0);
  }

  return normalized.slice(0, numTopics);
}

/**
 * Helper: Compute topic diversity (entropy)
 */
function computeTopicDiversity(posts) {
  if (posts.length === 0) return 0;

  const topics = posts.map(p => extractDominantTopic(p.content));
  const counts = {};
  topics.forEach(t => counts[t] = (counts[t] || 0) + 1);

  const probs = Object.values(counts).map(c => c / topics.length);
  const entropy = -probs.reduce((sum, p) => sum + (p * Math.log2(p)), 0);

  return normalize(entropy, 0, 5); // Normalize to 0-1
}

/**
 * Helper: Extract dominant topic (simplified: first hashtag or keyword)
 */
function extractDominantTopic(text) {
  const hashtags = text.match(/#\w+/g);
  if (hashtags && hashtags.length > 0) {
    return hashtags[0].replace('#', '');
  }
  
  // Fallback: first significant word
  const words = text.split(/\s+/).filter(w => w.length > 5);
  return words[0] || 'general';
}

/**
 * Helper: Compute recency score (exponential decay)
 */
function computeRecencyScore(events) {
  if (events.length === 0) return 0;

  const now = Date.now();
  const lambda = 0.1 / (24 * 60 * 60 * 1000); // 0.1 per day in ms

  const scores = events.map(e => {
    const age = now - new Date(e.timestamp).getTime();
    return Math.exp(-lambda * age);
  });

  return scores.reduce((a, b) => a + b, 0) / events.length;
}

/**
 * Helper: Compute consistency score (inverse of variance)
 */
function computeConsistencyScore(events) {
  if (events.length < 2) return 1;

  const timestamps = events.map(e => new Date(e.timestamp).getTime());
  timestamps.sort((a, b) => a - b);

  const intervals = [];
  for (let i = 1; i < timestamps.length; i++) {
    intervals.push(timestamps[i] - timestamps[i - 1]);
  }

  const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  const variance = intervals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / intervals.length;
  const std = Math.sqrt(variance);

  const consistency = 1 - Math.min(1, std / mean);
  return Math.max(0, consistency);
}

/**
 * Helper: Normalize value to 0-1 range
 */
function normalize(value, min, max) {
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

/**
 * Helper: Normalize vector to unit length
 */
function normalizeVector(vector) {
  const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  return magnitude > 0 ? vector.map(v => v / magnitude) : vector;
}

/**
 * Helper: Compute mode (most frequent value)
 */
function mode(arr) {
  const counts = {};
  arr.forEach(val => counts[val] = (counts[val] || 0) + 1);
  
  let maxCount = 0;
  let modeValue = 0;
  Object.entries(counts).forEach(([val, count]) => {
    if (count > maxCount) {
      maxCount = count;
      modeValue = parseFloat(val) || 0;
    }
  });
  
  return normalize(modeValue, 0, 100);
}

/**
 * Store computed embedding in database
 */
async function storeEngagementEmbedding(userId, embedding) {
  await pool.query(
    `INSERT INTO engagement_embeddings (user_id, embedding_vector, last_updated)
     VALUES ($1, $2, NOW())
     ON CONFLICT (user_id)
     DO UPDATE SET embedding_vector = $2, last_updated = NOW()`,
    [userId, JSON.stringify(embedding)]
  );
}

/**
 * Batch compute embeddings for all users (nightly job)
 */
async function computeAllEngagementEmbeddings() {
  try {
    const users = await pool.query('SELECT DISTINCT user_id FROM engagement_events');
    
    console.log(`Computing embeddings for ${users.rows.length} users...`);
    
    for (const { user_id } of users.rows) {
      try {
        const embedding = await computeEngagementEmbedding(user_id);
        await storeEngagementEmbedding(user_id, embedding);
      } catch (error) {
        console.error(`Failed to compute embedding for user ${user_id}:`, error);
      }
    }
    
    console.log('Batch embedding computation complete');
  } catch (error) {
    console.error('Error in batch computation:', error);
    throw error;
  }
}

module.exports = {
  computeEngagementEmbedding,
  storeEngagementEmbedding,
  computeAllEngagementEmbeddings,
};

