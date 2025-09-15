"""
Sentiment Analysis Module for Customer Feedback Analyzer

This module provides sentiment analysis capabilities using:
- TextBlob for basic sentiment analysis
- VADER (Valence Aware Dictionary and sEntiment Reasoner) for social media text
- Custom keyword-based sentiment scoring
- Emotion detection and classification

Author: Adyasha Khandai
Copyright (c) 2025 Adyasha Khandai
License: MIT License
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Import NLP libraries
from textblob import TextBlob
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VADER sentiment analyzer not available. Install with: pip install vaderSentiment")

from .text_preprocessor import TextPreprocessor


class SentimentAnalyzer:
    """Comprehensive sentiment analysis for customer feedback."""
    
    def __init__(self):
        """Initialize the sentiment analyzer with different models."""
        self.preprocessor = TextPreprocessor()
        
        # Initialize VADER if available
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            self.vader_analyzer = None
        
        # Define sentiment keywords for rule-based analysis
        self.positive_words = {
            'excellent', 'amazing', 'outstanding', 'fantastic', 'wonderful', 'perfect',
            'great', 'awesome', 'brilliant', 'superb', 'magnificent', 'exceptional',
            'love', 'loved', 'adore', 'impressed', 'satisfied', 'pleased', 'happy',
            'delighted', 'thrilled', 'grateful', 'recommend', 'recommending',
            'quality', 'fast', 'quick', 'efficient', 'helpful', 'friendly',
            'professional', 'reliable', 'trustworthy', 'responsive'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'disgusting', 'worst', 'hate', 'hated',
            'disappointed', 'frustrated', 'angry', 'annoyed', 'upset', 'dissatisfied',
            'poor', 'bad', 'useless', 'worthless', 'broken', 'defective',
            'slow', 'delayed', 'late', 'expensive', 'overpriced', 'rude',
            'unprofessional', 'unreliable', 'unresponsive', 'problem', 'issue',
            'complaint', 'refund', 'return', 'cancel', 'cancelled'
        }
        
        # Emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'delighted', 'pleased'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'irritated', 'frustrated'],
            'sadness': ['sad', 'disappointed', 'upset', 'depressed', 'unhappy'],
            'fear': ['worried', 'concerned', 'afraid', 'anxious', 'nervous'],
            'surprise': ['surprised', 'amazed', 'shocked', 'astonished'],
            'trust': ['trust', 'reliable', 'dependable', 'confident'],
            'anticipation': ['excited', 'eager', 'looking forward', 'anticipate']
        }
    
    def analyze_sentiment_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'label': 'neutral',
                'confidence': 0.0
            }
        
        # Preprocess text for sentiment analysis
        processed_text = self.preprocessor.preprocess_for_sentiment(text)
        
        # Create TextBlob object
        blob = TextBlob(processed_text)
        
        # Get polarity and subjectivity
        polarity = blob.sentiment.polarity  # Range: -1 (negative) to 1 (positive)
        subjectivity = blob.sentiment.subjectivity  # Range: 0 (objective) to 1 (subjective)
        
        # Determine sentiment label
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Calculate confidence based on absolute polarity value
        confidence = abs(polarity)
        
        return {
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'label': label,
            'confidence': round(confidence, 3),
            'method': 'textblob'
        }
    
    def analyze_sentiment_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with VADER sentiment analysis results
        """
        if not self.vader_analyzer or not text:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'label': 'neutral',
                'confidence': 0.0,
                'method': 'vader'
            }
        
        # VADER works well with raw text including punctuation and capitalization
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine label based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Use compound score as confidence measure
        confidence = abs(compound)
        
        return {
            'compound': round(compound, 3),
            'positive': round(scores['pos'], 3),
            'negative': round(scores['neg'], 3),
            'neutral': round(scores['neu'], 3),
            'label': label,
            'confidence': round(confidence, 3),
            'method': 'vader'
        }
    
    def analyze_sentiment_keywords(self, text: str) -> Dict:
        """
        Rule-based sentiment analysis using predefined keywords.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with keyword-based sentiment analysis results
        """
        if not text:
            return {
                'positive_count': 0,
                'negative_count': 0,
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.0,
                'method': 'keywords'
            }
        
        # Preprocess text
        tokens = self.preprocessor.preprocess_for_keywords(text)
        
        # Count positive and negative words
        positive_count = sum(1 for token in tokens if token.lower() in self.positive_words)
        negative_count = sum(1 for token in tokens if token.lower() in self.negative_words)
        
        # Calculate score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            score = 0.0
            label = 'neutral'
            confidence = 0.0
        else:
            score = (positive_count - negative_count) / total_sentiment_words
            if score > 0.2:
                label = 'positive'
            elif score < -0.2:
                label = 'negative'
            else:
                label = 'neutral'
            confidence = abs(score)
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'score': round(score, 3),
            'label': label,
            'confidence': round(confidence, 3),
            'method': 'keywords'
        }
    
    def detect_emotions(self, text: str) -> Dict:
        """
        Detect emotions in text using keyword matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with emotion detection results
        """
        if not text:
            return {}
        
        # Preprocess text
        tokens = self.preprocessor.preprocess_for_keywords(text)
        text_lower = text.lower()
        
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            # Count matches for this emotion
            count = 0
            for keyword in keywords:
                if keyword in text_lower:
                    count += 1
                # Also check individual tokens
                count += sum(1 for token in tokens if token == keyword)
            
            emotion_scores[emotion] = count
        
        # Normalize scores
        total_emotion_words = sum(emotion_scores.values())
        if total_emotion_words > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = round(emotion_scores[emotion] / total_emotion_words, 3)
        
        return emotion_scores
    
    def analyze_sentiment_comprehensive(self, text: str) -> Dict:
        """
        Comprehensive sentiment analysis using multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with combined sentiment analysis results
        """
        if not text:
            return {
                'overall_label': 'neutral',
                'overall_score': 0.0,
                'confidence': 0.0,
                'methods_used': []
            }
        
        results = {}
        
        # TextBlob analysis
        textblob_result = self.analyze_sentiment_textblob(text)
        results['textblob'] = textblob_result
        
        # VADER analysis (if available)
        if self.vader_analyzer:
            vader_result = self.analyze_sentiment_vader(text)
            results['vader'] = vader_result
        
        # Keyword-based analysis
        keywords_result = self.analyze_sentiment_keywords(text)
        results['keywords'] = keywords_result
        
        # Emotion detection
        emotions = self.detect_emotions(text)
        results['emotions'] = emotions
        
        # Combine results to get overall sentiment
        overall = self._combine_sentiment_results(results)
        results['overall'] = overall
        
        return results
    
    def _combine_sentiment_results(self, results: Dict) -> Dict:
        """
        Combine results from different sentiment analysis methods.
        
        Args:
            results: Dictionary containing results from different methods
            
        Returns:
            Dictionary with combined/overall sentiment
        """
        scores = []
        labels = []
        confidences = []
        methods_used = []
        
        # Collect scores and labels from different methods
        if 'textblob' in results:
            scores.append(results['textblob']['polarity'])
            labels.append(results['textblob']['label'])
            confidences.append(results['textblob']['confidence'])
            methods_used.append('textblob')
        
        if 'vader' in results:
            scores.append(results['vader']['compound'])
            labels.append(results['vader']['label'])
            confidences.append(results['vader']['confidence'])
            methods_used.append('vader')
        
        if 'keywords' in results:
            scores.append(results['keywords']['score'])
            labels.append(results['keywords']['label'])
            confidences.append(results['keywords']['confidence'])
            methods_used.append('keywords')
        
        if not scores:
            return {
                'overall_label': 'neutral',
                'overall_score': 0.0,
                'confidence': 0.0,
                'methods_used': []
            }
        
        # Calculate overall score (weighted average)
        weights = confidences if any(c > 0 for c in confidences) else [1] * len(scores)
        overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Determine overall label
        if overall_score > 0.1:
            overall_label = 'positive'
        elif overall_score < -0.1:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        # Calculate overall confidence
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'overall_label': overall_label,
            'overall_score': round(overall_score, 3),
            'confidence': round(overall_confidence, 3),
            'methods_used': methods_used,
            'individual_scores': scores,
            'individual_labels': labels
        }
    
    def extract_keywords_with_sentiment(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords with their sentiment relevance scores.
        
        Args:
            text: Text to analyze
            top_n: Number of top keywords to return
            
        Returns:
            List of tuples (keyword, relevance_score)
        """
        if not text:
            return []
        
        # Get processed tokens
        tokens = self.preprocessor.preprocess_for_keywords(text)
        
        # Get key phrases
        phrases = self.preprocessor.extract_key_phrases(text, max_phrases=top_n * 2)
        
        # Combine tokens and phrases
        candidates = tokens + phrases
        
        # Calculate relevance scores based on sentiment and frequency
        keyword_scores = {}
        word_counts = Counter(candidates)
        
        for word, frequency in word_counts.items():
            # Base score from frequency
            score = frequency / len(candidates)
            
            # Boost score if word has sentiment value
            if word.lower() in self.positive_words:
                score *= 1.5
            elif word.lower() in self.negative_words:
                score *= 1.3
            
            # Boost score for longer phrases (usually more specific)
            if ' ' in word:
                score *= 1.2
            
            keyword_scores[word] = score
        
        # Sort by score and return top N
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]
    
    def analyze_feedback_batch(self, feedback_list: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a batch of feedback items.
        
        Args:
            feedback_list: List of feedback dictionaries with 'feedback_text' key
            
        Returns:
            List of feedback dictionaries with added sentiment analysis
        """
        analyzed_feedback = []
        
        for feedback in feedback_list:
            feedback_text = feedback.get('feedback_text', '')
            
            # Perform comprehensive sentiment analysis
            sentiment_results = self.analyze_sentiment_comprehensive(feedback_text)
            
            # Extract keywords
            keywords = self.extract_keywords_with_sentiment(feedback_text, top_n=5)
            
            # Add analysis results to feedback
            analyzed_item = feedback.copy()
            analyzed_item['sentiment_analysis'] = sentiment_results
            analyzed_item['extracted_keywords'] = keywords
            
            analyzed_feedback.append(analyzed_item)
        
        return analyzed_feedback


# Example usage and testing
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test samples
    test_samples = [
        "This product is absolutely amazing! I love it and would definitely recommend it to others.",
        "Terrible quality and poor customer service. Very disappointed with this purchase.",
        "The product is okay, nothing special but does the job.",
        "Fast delivery and good quality! However, the price is a bit high.",
        "I'm so frustrated with this service. It's unreliable and the support team is unhelpful."
    ]
    
    print("Sentiment Analysis Test Results:")
    print("=" * 50)
    
    for i, text in enumerate(test_samples, 1):
        print(f"\\nSample {i}: {text}")
        
        # Comprehensive analysis
        results = analyzer.analyze_sentiment_comprehensive(text)
        overall = results['overall']
        
        print(f"Overall Sentiment: {overall['overall_label']} (Score: {overall['overall_score']}, Confidence: {overall['confidence']})")
        
        # Keywords
        keywords = analyzer.extract_keywords_with_sentiment(text, top_n=3)
        print(f"Top Keywords: {[kw[0] for kw in keywords]}")
        
        print("-" * 30)