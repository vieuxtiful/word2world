/**
 * useEngagementTracking.jsx
 * 
 * React hook for tracking user engagement events on the Word2World platform.
 * Tracks: views, likes, comments, saves, and hashtags.
 * 
 * Usage:
 *   import useEngagementTracking from './hooks/useEngagementTracking';
 *   
 *   function PostComponent({ post }) {
 *     const { trackView, trackLike, trackComment, trackSave, trackHashtag } = useEngagementTracking();
 *     
 *     return (
 *       <div ref={(el) => trackView(el, post.id)}>
 *         <button onClick={() => trackLike(post.id)}>Like</button>
 *       </div>
 *     );
 *   }
 */

import { useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000/api';

const useEngagementTracking = () => {
  const viewTimers = useRef(new Map());
  const viewStartTimes = useRef(new Map());
  const observerRef = useRef(null);

  /**
   * Log engagement event to backend
   */
  const logEngagementEvent = useCallback(async (eventType, eventData) => {
    try {
      const userId = localStorage.getItem('user_id'); // Adjust based on your auth system
      
      if (!userId) {
        console.warn('User not authenticated, skipping engagement tracking');
        return;
      }

      await axios.post(`${API_BASE_URL}/engagement/log`, {
        user_id: userId,
        event_type: eventType,
        event_data: {
          ...eventData,
          timestamp: new Date().toISOString(),
        },
      });
    } catch (error) {
      console.error('Failed to log engagement event:', error);
      // Optionally queue for retry
    }
  }, []);

  /**
   * Track view events using Intersection Observer
   * Post is considered "viewed" if visible > 50% for > 2 seconds
   */
  const trackView = useCallback((element, postId) => {
    if (!element || !postId) return;

    // Initialize Intersection Observer if not already created
    if (!observerRef.current) {
      observerRef.current = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            const postId = entry.target.dataset.postId;
            
            if (entry.isIntersecting && entry.intersectionRatio > 0.5) {
              // Post entered viewport with > 50% visibility
              const startTime = Date.now();
              viewStartTimes.current.set(postId, startTime);
              
              // Set timer to log view after 2 seconds
              const timer = setTimeout(() => {
                const duration = Date.now() - startTime;
                const scrollDepth = window.scrollY / (document.body.scrollHeight - window.innerHeight);
                
                logEngagementEvent('view', {
                  post_id: postId,
                  duration_ms: duration,
                  scroll_depth: Math.min(1, Math.max(0, scrollDepth)),
                  viewport_ratio: entry.intersectionRatio,
                });
                
                viewTimers.current.delete(postId);
                viewStartTimes.current.delete(postId);
              }, 2000);
              
              viewTimers.current.set(postId, timer);
            } else {
              // Post left viewport or visibility dropped below 50%
              const timer = viewTimers.current.get(postId);
              if (timer) {
                clearTimeout(timer);
                viewTimers.current.delete(postId);
                viewStartTimes.current.delete(postId);
              }
            }
          });
        },
        {
          threshold: [0, 0.25, 0.5, 0.75, 1.0],
          rootMargin: '0px',
        }
      );
    }

    // Set post ID as data attribute and observe
    element.dataset.postId = postId;
    observerRef.current.observe(element);

    // Cleanup function
    return () => {
      if (observerRef.current && element) {
        observerRef.current.unobserve(element);
      }
      const timer = viewTimers.current.get(postId);
      if (timer) {
        clearTimeout(timer);
        viewTimers.current.delete(postId);
      }
    };
  }, [logEngagementEvent]);

  /**
   * Track like events
   */
  const trackLike = useCallback(async (postId) => {
    await logEngagementEvent('like', {
      post_id: postId,
    });
  }, [logEngagementEvent]);

  /**
   * Track comment events
   */
  const trackComment = useCallback(async (postId, commentText, parentCommentId = null) => {
    await logEngagementEvent('comment', {
      post_id: postId,
      comment_text: commentText,
      comment_length: commentText.length,
      parent_comment_id: parentCommentId,
      reply_depth: parentCommentId ? 1 : 0, // Simplified; can be calculated recursively
    });
  }, [logEngagementEvent]);

  /**
   * Track save/bookmark events
   */
  const trackSave = useCallback(async (postId) => {
    await logEngagementEvent('save', {
      post_id: postId,
    });
  }, [logEngagementEvent]);

  /**
   * Track hashtag usage events
   */
  const trackHashtag = useCallback(async (hashtag, postId = null) => {
    await logEngagementEvent('hashtag', {
      hashtag: hashtag.replace('#', ''), // Remove # prefix if present
      post_id: postId,
    });
  }, [logEngagementEvent]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      // Clear all pending view timers
      viewTimers.current.forEach((timer) => clearTimeout(timer));
      viewTimers.current.clear();
      viewStartTimes.current.clear();
      
      // Disconnect observer
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, []);

  return {
    trackView,
    trackLike,
    trackComment,
    trackSave,
    trackHashtag,
  };
};

export default useEngagementTracking;

