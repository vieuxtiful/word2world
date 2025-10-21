/**
 * server.js
 * 
 * Main Express.js server for Word2World engagement tracking backend.
 * 
 * Features:
 * - Engagement event logging (views, likes, comments, saves, hashtags)
 * - Engagement embedding computation
 * - Scheduled batch processing
 * - RESTful API endpoints
 * 
 * Usage:
 *   npm install
 *   npm start
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const cron = require('node-cron');

// Import routes and services
const engagementRoutes = require('./engagementRoutes');
const { computeAllEngagementEmbeddings } = require('./engagementEmbeddingService');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet()); // Security headers
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3001',
  credentials: true,
}));
app.use(morgan('combined')); // Logging
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'word2world-engagement-api',
  });
});

// API routes
app.use('/api/engagement', engagementRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    success: false,
    error: err.message || 'Internal server error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
  });
});

// Scheduled jobs
// Run nightly at 2:00 AM to compute all engagement embeddings
cron.schedule('0 2 * * *', async () => {
  console.log('[CRON] Starting nightly engagement embedding computation...');
  try {
    await computeAllEngagementEmbeddings();
    console.log('[CRON] Nightly embedding computation completed successfully');
  } catch (error) {
    console.error('[CRON] Error in nightly embedding computation:', error);
  }
}, {
  timezone: 'America/New_York', // Adjust to your timezone
});

// Refresh materialized view every 6 hours
cron.schedule('0 */6 * * *', async () => {
  console.log('[CRON] Refreshing daily engagement aggregates...');
  const { Pool } = require('pg');
  const pool = new Pool({
    host: process.env.DB_HOST || 'localhost',
    port: process.env.DB_PORT || 5432,
    database: process.env.DB_NAME || 'word2world',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD,
  });
  
  try {
    await pool.query('SELECT refresh_daily_aggregates()');
    console.log('[CRON] Daily aggregates refreshed successfully');
  } catch (error) {
    console.error('[CRON] Error refreshing aggregates:', error);
  } finally {
    await pool.end();
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Word2World Engagement Tracking API                         ║
║   Server running on port ${PORT}                              ║
║   Environment: ${process.env.NODE_ENV || 'development'}                                    ║
║                                                               ║
║   Endpoints:                                                  ║
║   - POST /api/engagement/log                                  ║
║   - POST /api/engagement/batch                                ║
║   - GET  /api/engagement/stats/:userId                        ║
║   - GET  /health                                              ║
║                                                               ║
║   Scheduled Jobs:                                             ║
║   - Nightly embedding computation: 2:00 AM daily              ║
║   - Aggregate refresh: Every 6 hours                          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
  `);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM signal received: closing HTTP server');
  app.close(() => {
    console.log('HTTP server closed');
  });
});

module.exports = app; // For testing

