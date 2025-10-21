/**
 * PostCard.jsx
 * 
 * Example React component demonstrating usage of useEngagementTracking hook.
 * Displays a social media post with engagement tracking for views, likes, 
 * comments, saves, and hashtags.
 */

import React, { useState, useRef, useEffect } from 'react';
import useEngagementTracking from './hooks/useEngagementTracking';
import './PostCard.css';

const PostCard = ({ post }) => {
  const [isLiked, setIsLiked] = useState(post.isLiked || false);
  const [isSaved, setIsSaved] = useState(post.isSaved || false);
  const [commentText, setCommentText] = useState('');
  const [showComments, setShowComments] = useState(false);
  
  const postRef = useRef(null);
  const { trackView, trackLike, trackComment, trackSave, trackHashtag } = useEngagementTracking();

  // Track view when component mounts
  useEffect(() => {
    if (postRef.current) {
      const cleanup = trackView(postRef.current, post.id);
      return cleanup;
    }
  }, [post.id, trackView]);

  // Handle like
  const handleLike = () => {
    setIsLiked(!isLiked);
    if (!isLiked) {
      trackLike(post.id);
    }
  };

  // Handle save/bookmark
  const handleSave = () => {
    setIsSaved(!isSaved);
    if (!isSaved) {
      trackSave(post.id);
    }
  };

  // Handle comment submission
  const handleCommentSubmit = (e) => {
    e.preventDefault();
    if (commentText.trim()) {
      trackComment(post.id, commentText);
      
      // Extract and track hashtags from comment
      const hashtags = commentText.match(/#\w+/g);
      if (hashtags) {
        hashtags.forEach(tag => trackHashtag(tag, post.id));
      }
      
      setCommentText('');
      // Add comment to UI (would typically update state/refetch)
    }
  };

  // Track hashtags in post content on mount
  useEffect(() => {
    const hashtags = post.content.match(/#\w+/g);
    if (hashtags) {
      // Only track when user interacts with hashtag (click)
      // This is just for demonstration
    }
  }, [post.content]);

  return (
    <div className="post-card" ref={postRef} data-post-id={post.id}>
      {/* Post Header */}
      <div className="post-header">
        <img 
          src={post.author.avatar} 
          alt={post.author.name}
          className="author-avatar"
        />
        <div className="author-info">
          <h3>{post.author.name}</h3>
          <span className="post-time">{post.timestamp}</span>
        </div>
      </div>

      {/* Post Content */}
      <div className="post-content">
        <p>{post.content}</p>
        {post.image && (
          <img src={post.image} alt="Post content" className="post-image" />
        )}
      </div>

      {/* Engagement Actions */}
      <div className="post-actions">
        <button 
          className={`action-btn ${isLiked ? 'liked' : ''}`}
          onClick={handleLike}
          aria-label="Like post"
        >
          <span className="icon">❤️</span>
          <span>{post.likeCount + (isLiked ? 1 : 0)}</span>
        </button>

        <button 
          className="action-btn"
          onClick={() => setShowComments(!showComments)}
          aria-label="Comment on post"
        >
          <span className="icon">💬</span>
          <span>{post.commentCount}</span>
        </button>

        <button 
          className={`action-btn ${isSaved ? 'saved' : ''}`}
          onClick={handleSave}
          aria-label="Save post"
        >
          <span className="icon">🔖</span>
        </button>

        <button 
          className="action-btn"
          aria-label="Share post"
        >
          <span className="icon">🔗</span>
        </button>
      </div>

      {/* Comments Section */}
      {showComments && (
        <div className="comments-section">
          <form onSubmit={handleCommentSubmit} className="comment-form">
            <input
              type="text"
              value={commentText}
              onChange={(e) => setCommentText(e.target.value)}
              placeholder="Write a comment... (use #hashtags)"
              className="comment-input"
            />
            <button type="submit" className="comment-submit">
              Post
            </button>
          </form>

          {/* Existing comments would be rendered here */}
          <div className="comments-list">
            {post.comments?.map(comment => (
              <div key={comment.id} className="comment">
                <strong>{comment.author}</strong>: {comment.text}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PostCard;

