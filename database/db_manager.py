"""
Database Manager for Customer Feedback Analyzer

This module handles all database operations including:
- Database connection and initialization
- CRUD operations for feedback data
- Storing and retrieving analysis results
- Data validation and error handling

Author: Adyasha Khandai
Copyright (c) 2025 Adyasha Khandai
License: MIT License
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class DatabaseManager:
    """Manages SQLite database operations for the feedback analyzer."""
    
    def __init__(self, db_path: str = "feedback_analyzer.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def init_database(self):
        """Initialize database with schema from schema.sql file."""
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        
        with self.get_connection() as conn:
            with open(schema_path, 'r') as schema_file:
                schema_sql = schema_file.read()
                conn.executescript(schema_sql)
            conn.commit()
    
    def add_feedback(self, customer_name: str, email: str, product_category: str, 
                    rating: int, feedback_text: str) -> int:
        """
        Add new feedback to the database.
        
        Args:
            customer_name: Name of the customer
            email: Customer's email address
            product_category: Category of the product/service
            rating: Rating from 1-5
            feedback_text: The actual feedback text
            
        Returns:
            The ID of the newly inserted feedback record
            
        Raises:
            ValueError: If rating is not between 1-5
            sqlite3.Error: If database operation fails
        """
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")
        
        if not feedback_text.strip():
            raise ValueError("Feedback text cannot be empty")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (customer_name, email, product_category, rating, feedback_text)
                VALUES (?, ?, ?, ?, ?)
            """, (customer_name, email, product_category, rating, feedback_text))
            conn.commit()
            return cursor.lastrowid
    
    def get_all_feedback(self) -> List[Dict]:
        """
        Retrieve all feedback from the database.
        
        Returns:
            List of dictionaries containing feedback data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, customer_name, email, product_category, 
                       rating, feedback_text, created_date
                FROM feedback
                ORDER BY created_date DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_feedback_by_category(self, category: str) -> List[Dict]:
        """
        Get feedback filtered by product category.
        
        Args:
            category: Product category to filter by
            
        Returns:
            List of feedback dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, customer_name, email, product_category, 
                       rating, feedback_text, created_date
                FROM feedback
                WHERE product_category = ?
                ORDER BY created_date DESC
            """, (category,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_feedback_by_rating(self, min_rating: int = 1, max_rating: int = 5) -> List[Dict]:
        """
        Get feedback filtered by rating range.
        
        Args:
            min_rating: Minimum rating to include
            max_rating: Maximum rating to include
            
        Returns:
            List of feedback dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, customer_name, email, product_category, 
                       rating, feedback_text, created_date
                FROM feedback
                WHERE rating BETWEEN ? AND ?
                ORDER BY created_date DESC
            """, (min_rating, max_rating))
            return [dict(row) for row in cursor.fetchall()]
    
    def save_sentiment_analysis(self, feedback_id: int, sentiment_score: float, 
                               sentiment_label: str, confidence: float = None) -> int:
        """
        Save sentiment analysis results to the database.
        
        Args:
            feedback_id: ID of the feedback being analyzed
            sentiment_score: Sentiment score (-1 to 1)
            sentiment_label: Sentiment label ('positive', 'negative', 'neutral')
            confidence: Confidence score of the analysis
            
        Returns:
            ID of the inserted sentiment analysis record
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sentiment_analysis (feedback_id, sentiment_score, sentiment_label, confidence)
                VALUES (?, ?, ?, ?)
            """, (feedback_id, sentiment_score, sentiment_label, confidence))
            conn.commit()
            return cursor.lastrowid
    
    def save_keywords(self, feedback_id: int, keywords: List[Tuple[str, int, float]]):
        """
        Save extracted keywords for a feedback.
        
        Args:
            feedback_id: ID of the feedback
            keywords: List of tuples (keyword, frequency, relevance_score)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO keywords (feedback_id, keyword, frequency, relevance_score)
                VALUES (?, ?, ?, ?)
            """, [(feedback_id, kw[0], kw[1], kw[2]) for kw in keywords])
            conn.commit()
    
    def get_sentiment_summary(self) -> Dict:
        """
        Get summary statistics of sentiment analysis.
        
        Returns:
            Dictionary containing sentiment distribution
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT sentiment_label, COUNT(*) as count,
                       AVG(sentiment_score) as avg_score
                FROM sentiment_analysis
                GROUP BY sentiment_label
            """)
            
            results = cursor.fetchall()
            summary = {}
            for row in results:
                summary[row['sentiment_label']] = {
                    'count': row['count'],
                    'avg_score': round(row['avg_score'], 3)
                }
            
            # Get total count
            cursor.execute("SELECT COUNT(*) as total FROM sentiment_analysis")
            total = cursor.fetchone()['total']
            summary['total'] = total
            
            return summary
    
    def get_top_keywords(self, limit: int = 10) -> List[Dict]:
        """
        Get the most frequent keywords across all feedback.
        
        Args:
            limit: Maximum number of keywords to return
            
        Returns:
            List of dictionaries with keyword statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT keyword, 
                       COUNT(*) as total_occurrences,
                       AVG(frequency) as avg_frequency,
                       AVG(relevance_score) as avg_relevance
                FROM keywords
                GROUP BY keyword
                ORDER BY total_occurrences DESC, avg_relevance DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_feedback_with_sentiment(self) -> List[Dict]:
        """
        Get all feedback with their corresponding sentiment analysis.
        
        Returns:
            List of dictionaries containing feedback and sentiment data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.id, f.customer_name, f.email, f.product_category,
                       f.rating, f.feedback_text, f.created_date,
                       s.sentiment_score, s.sentiment_label, s.confidence
                FROM feedback f
                LEFT JOIN sentiment_analysis s ON f.id = s.feedback_id
                ORDER BY f.created_date DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_feedback(self, feedback_id: int) -> bool:
        """
        Delete feedback and associated analysis results.
        
        Args:
            feedback_id: ID of the feedback to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete associated records first (foreign key constraints)
            cursor.execute("DELETE FROM keywords WHERE feedback_id = ?", (feedback_id,))
            cursor.execute("DELETE FROM sentiment_analysis WHERE feedback_id = ?", (feedback_id,))
            cursor.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_database_stats(self) -> Dict:
        """
        Get overall database statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total feedback count
            cursor.execute("SELECT COUNT(*) as count FROM feedback")
            stats['total_feedback'] = cursor.fetchone()['count']
            
            # Average rating
            cursor.execute("SELECT AVG(rating) as avg_rating FROM feedback")
            result = cursor.fetchone()
            stats['avg_rating'] = round(result['avg_rating'], 2) if result['avg_rating'] else 0
            
            # Category distribution
            cursor.execute("""
                SELECT product_category, COUNT(*) as count 
                FROM feedback 
                GROUP BY product_category 
                ORDER BY count DESC
            """)
            stats['category_distribution'] = {row['product_category']: row['count'] 
                                            for row in cursor.fetchall()}
            
            # Analysis completion rate
            cursor.execute("SELECT COUNT(*) as count FROM sentiment_analysis")
            analyzed_count = cursor.fetchone()['count']
            stats['analysis_completion_rate'] = (
                round((analyzed_count / stats['total_feedback']) * 100, 1) 
                if stats['total_feedback'] > 0 else 0
            )
            
            return stats


# Example usage and testing functions
if __name__ == "__main__":
    # Test the database manager
    db = DatabaseManager("test_feedback.db")
    
    # Add sample feedback
    feedback_id = db.add_feedback(
        customer_name="John Doe",
        email="john@example.com",
        product_category="Electronics",
        rating=4,
        feedback_text="Great product! Really satisfied with the quality and fast delivery."
    )
    
    print(f"Added feedback with ID: {feedback_id}")
    
    # Get all feedback
    all_feedback = db.get_all_feedback()
    print(f"Total feedback records: {len(all_feedback)}")
    
    # Get database stats
    stats = db.get_database_stats()
    print("Database Statistics:", stats)