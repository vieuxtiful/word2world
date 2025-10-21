/**
 * engagementRoutes.js
 * 
 * Express.js routes for logging and managing user engagement events.
 * Handles: views, likes, comments, saves, and hashtags.
 * 
 * Usage:
 *   const engagementRoutes = require('./routes/engagementRoutes');
 *   app.use('/api/engagement', engagementRoutes);
 */

const express = require('express');
const router = express.Router();
const { Pool } = require('pg');
const { body, validationResult } = require('express-validator');

// Database connection pool
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'word2world',
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASSWORD,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Middleware to verify authentication (adjust based on your auth system)
const authenticateUser = (req, res, next) => {
  const userId = req.body.user_id || req.headers['x-user-id'];
  
  if (!userId) {
    return res.status(401).json({ 
      success: false, 
      error: 'Authentication required' 
    });
  }
  
  req.userId = userId;
  next();
};

/**
 * POST /api/engagement/log
 * Log a single engagement event
 */
router.post('/log', 
  authenticateUser,
  [
    body('event_type').isIn(['view', 'like', 'comment', 'save', 'hashtag']),
    body('event_data').isObject(),
  ],
  async (req, res) => {
    // Validate request
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ 
        success: false, 
        errors: errors.array() 
      });
    }

    const { event_type, event_data } = req.body;
    const userId = req.userId;

    try {
      // Insert engagement event
      const result = await pool.query(
        `INSERT INTO engagement_events 
         (user_id, event_type, event_data, timestamp) 
         VALUES ($1, $2, $3, NOW()) 
         RETURNING id`,
        [userId, event_type, JSON.stringify(event_data)]
      );

      // Check if user is active (> 10 events/day)
      const isActive = await checkIfActiveUser(userId);
      
      // Queue incremental embedding update for active users
      if (isActive) {
        await queueEmbeddingUpdate(userId);
      }

      res.json({ 
        success: true, 
        event_id: result.rows[0].id 
      });
    } catch (error) {
      console.error('Error logging engagement event:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to log engagement event' 
      });
    }
  }
);

/**
 * POST /api/engagement/batch
 * Log multiple engagement events in batch
 */
router.post('/batch',
  authenticateUser,
  [
    body('events').isArray(),
    body('events.*.event_type').isIn(['view', 'like', 'comment', 'save', 'hashtag']),
    body('events.*.event_data').isObject(),
  ],
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ 
        success: false, 
        errors: errors.array() 
      });
    }

    const { events } = req.body;
    const userId = req.userId;

    try {
      // Build batch insert query
      const values = events.map((event, idx) => {
        const offset = idx * 3;
        return `($${offset + 1}, $${offset + 2}, $${offset + 3}, NOW())`;
      }).join(', ');

      const params = events.flatMap(event => [
        userId,
        event.event_type,
        JSON.stringify(event.event_data)
      ]);

      await pool.query(
        `INSERT INTO engagement_events 
         (user_id, event_type, event_data, timestamp) 
         VALUES ${values}`,
        params
      );

      res.json({ 
        success: true, 
        count: events.length 
      });
    } catch (error) {
      console.error('Error logging batch engagement events:', error);
      res.status(500).json({ 
        success: false, 
        error: 'Failed to log batch events' 
      });
    }
  }
);

/**
 * GET /api/engagement/stats/:userId
 * Get engagement statistics for a user
 */
router.get('/stats/:userId', async (req, res) => {
  const { userId } = req.params;
  const { timeRange = '30d' } = req.query;

  try {
    // Parse time range
    const timeRangeMap = {
      '24h': '1 day',
      '7d': '7 days',
      '30d': '30 days',
      '90d': '90 days',
    };
    const interval = timeRangeMap[timeRange] || '30 days';

    // Get engagement counts by type
    const stats = await pool.query(
      `SELECT 
         event_type,
         COUNT(*) as count,
         MIN(timestamp) as first_event,
         MAX(timestamp) as last_event
       FROM engagement_events
       WHERE user_id = $1 
         AND timestamp > NOW() - INTERVAL '${interval}'
       GROUP BY event_type`,
      [userId]
    );

    // Get engagement embedding if available
    const embedding = await pool.query(
      `SELECT embedding_vector, last_updated, feature_metadata
       FROM engagement_embeddings
       WHERE user_id = $1`,
      [userId]
    );

    res.json({
      success: true,
      stats: stats.rows,
      embedding: embedding.rows[0] || null,
    });
  } catch (error) {
    console.error('Error fetching engagement stats:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to fetch stats' 
    });
  }
});

/**
 * Helper: Check if user is active (> 10 events/day)
 */
async function checkIfActiveUser(userId) {
  try {
    const result = await pool.query(
      `SELECT COUNT(*) as count
       FROM engagement_events
       WHERE user_id = $1 
         AND timestamp > NOW() - INTERVAL '1 day'`,
      [userId]
    );
    
    return parseInt(result.rows[0].count) > 10;
  } catch (error) {
    console.error('Error checking active user:', error);
    return false;
  }
}

/**
 * Helper: Queue embedding update for user
 * (In production, this would use a job queue like Bull or AWS SQS)
 */
async function queueEmbeddingUpdate(userId) {
  try {
    // Simple implementation: insert into update queue table
    await pool.query(
      `INSERT INTO embedding_update_queue (user_id, queued_at, status)
       VALUES ($1, NOW(), 'pending')
       ON CONFLICT (user_id) 
       DO UPDATE SET queued_at = NOW(), status = 'pending'`,
      [userId]
    );
  } catch (error) {
    console.error('Error queueing embedding update:', error);
  }
}

module.exports = router;

