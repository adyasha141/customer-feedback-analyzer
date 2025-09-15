"""
System Test Script for Customer Feedback Analyzer

This script tests the main components of the system to ensure
everything is working correctly.

Author: Adyasha Khandai
Copyright (c) 2025 Adyasha Khandai
License: MIT License
"""

import os
import sys
import tempfile
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from database.db_manager import DatabaseManager
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.text_preprocessor import TextPreprocessor
from visualization.charts import FeedbackVisualizer


def test_database_operations():
    """Test database operations."""
    print("🧪 Testing Database Operations...")
    
    try:
        # Use a temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
        
        # Initialize database
        db = DatabaseManager(temp_db_path)
        
        # Test adding feedback
        feedback_id = db.add_feedback(
            customer_name="Test User",
            email="test@example.com",
            product_category="Electronics",
            rating=4,
            feedback_text="This is a test feedback for the system."
        )
        
        assert feedback_id > 0, "Failed to add feedback"
        print("  ✅ Add feedback: PASSED")
        
        # Test retrieving feedback
        all_feedback = db.get_all_feedback()
        assert len(all_feedback) == 1, "Failed to retrieve feedback"
        assert all_feedback[0]['customer_name'] == "Test User", "Incorrect feedback data"
        print("  ✅ Retrieve feedback: PASSED")
        
        # Test database statistics
        stats = db.get_database_stats()
        assert stats['total_feedback'] == 1, "Incorrect statistics"
        assert stats['avg_rating'] == 4.0, "Incorrect average rating"
        print("  ✅ Database statistics: PASSED")
        
        # Cleanup
        os.unlink(temp_db_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        traceback.print_exc()
        return False


def test_text_preprocessing():
    """Test text preprocessing functionality."""
    print("\\n🧪 Testing Text Preprocessing...")
    
    try:
        preprocessor = TextPreprocessor()
        
        # Test text cleaning
        test_text = "  This is a TEST with URLS http://example.com and extra   spaces!  "
        cleaned = preprocessor.clean_text(test_text)
        assert "http://example.com" not in cleaned, "URL not removed"
        assert cleaned.strip() == cleaned, "Extra spaces not removed"
        print("  ✅ Text cleaning: PASSED")
        
        # Test tokenization
        tokens = preprocessor.tokenize("Hello world! This is a test.")
        assert len(tokens) > 0, "Tokenization failed"
        assert "Hello" in [t.title() for t in tokens], "Incorrect tokenization"
        print("  ✅ Tokenization: PASSED")
        
        # Test keyword extraction
        test_feedback = "This product is amazing! Great quality and fast delivery. Highly recommended!"
        keywords = preprocessor.preprocess_for_keywords(test_feedback)
        assert len(keywords) > 0, "Keyword extraction failed"
        print("  ✅ Keyword extraction: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Text preprocessing test failed: {e}")
        traceback.print_exc()
        return False


def test_sentiment_analysis():
    """Test sentiment analysis functionality."""
    print("\\n🧪 Testing Sentiment Analysis...")
    
    try:
        analyzer = SentimentAnalyzer()
        
        # Test positive sentiment
        positive_text = "I absolutely love this product! It's amazing and exceeded my expectations."
        result = analyzer.analyze_sentiment_comprehensive(positive_text)
        assert 'overall' in result, "Missing overall sentiment"
        assert result['overall']['overall_label'] in ['positive', 'negative', 'neutral'], "Invalid sentiment label"
        print("  ✅ Positive sentiment analysis: PASSED")
        
        # Test negative sentiment
        negative_text = "This product is terrible! Worst purchase ever. Complete waste of money."
        result = analyzer.analyze_sentiment_comprehensive(negative_text)
        assert result['overall']['overall_label'] in ['positive', 'negative', 'neutral'], "Invalid sentiment label"
        print("  ✅ Negative sentiment analysis: PASSED")
        
        # Test keyword extraction with sentiment
        keywords = analyzer.extract_keywords_with_sentiment(positive_text, top_n=3)
        assert len(keywords) > 0, "Keyword extraction failed"
        assert all(len(kw) == 2 for kw in keywords), "Incorrect keyword format"
        print("  ✅ Keyword extraction with sentiment: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sentiment analysis test failed: {e}")
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization functionality."""
    print("\\n🧪 Testing Visualization...")
    
    try:
        visualizer = FeedbackVisualizer()
        
        # Create sample data
        sample_data = [
            {
                'id': 1,
                'customer_name': 'John Doe',
                'product_category': 'Electronics',
                'rating': 5,
                'feedback_text': 'Great product!',
                'sentiment_label': 'positive'
            },
            {
                'id': 2,
                'customer_name': 'Jane Smith',
                'product_category': 'Clothing',
                'rating': 2,
                'feedback_text': 'Poor quality.',
                'sentiment_label': 'negative'
            }
        ]
        
        # Test sentiment distribution plot
        fig = visualizer.plot_sentiment_distribution(sample_data, show_plot=False)
        assert fig is not None, "Failed to create sentiment distribution plot"
        print("  ✅ Sentiment distribution plot: PASSED")
        
        # Test rating distribution plot
        fig = visualizer.plot_rating_distribution(sample_data, show_plot=False)
        assert fig is not None, "Failed to create rating distribution plot"
        print("  ✅ Rating distribution plot: PASSED")
        
        # Test category analysis plot
        fig = visualizer.plot_category_analysis(sample_data, show_plot=False)
        assert fig is not None, "Failed to create category analysis plot"
        print("  ✅ Category analysis plot: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualization test failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components."""
    print("\\n🧪 Testing Component Integration...")
    
    try:
        # Use temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            temp_db_path = temp_db.name
        
        # Initialize all components
        db = DatabaseManager(temp_db_path)
        analyzer = SentimentAnalyzer()
        
        # Add sample feedback
        feedback_id = db.add_feedback(
            customer_name="Integration Test",
            email="test@integration.com",
            product_category="Software",
            rating=5,
            feedback_text="This software is absolutely fantastic! It works perfectly and saves me so much time."
        )
        
        # Analyze the feedback
        feedback_data = db.get_all_feedback()
        feedback_text = feedback_data[0]['feedback_text']
        
        results = analyzer.analyze_sentiment_comprehensive(feedback_text)
        keywords = analyzer.extract_keywords_with_sentiment(feedback_text, top_n=3)
        
        # Save analysis results
        db.save_sentiment_analysis(
            feedback_id=feedback_id,
            sentiment_score=results['overall']['overall_score'],
            sentiment_label=results['overall']['overall_label'],
            confidence=results['overall']['confidence']
        )
        
        keyword_tuples = [(kw[0], 1, kw[1]) for kw in keywords]
        if keyword_tuples:
            db.save_keywords(feedback_id, keyword_tuples)
        
        # Verify integration
        feedback_with_sentiment = db.get_feedback_with_sentiment()
        assert len(feedback_with_sentiment) == 1, "Integration failed"
        assert feedback_with_sentiment[0]['sentiment_label'] is not None, "Sentiment not saved"
        
        print("  ✅ Database + Sentiment Analysis integration: PASSED")
        
        # Cleanup
        os.unlink(temp_db_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all system tests."""
    print("🧪 Customer Feedback Analyzer - System Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run individual tests
    test_results.append(("Database Operations", test_database_operations()))
    test_results.append(("Text Preprocessing", test_text_preprocessing()))
    test_results.append(("Sentiment Analysis", test_sentiment_analysis()))
    test_results.append(("Visualization", test_visualization()))
    test_results.append(("Integration", test_integration()))
    
    # Display results
    print("\\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\\n🎉 All tests passed! System is working correctly.")
        print("💡 You can now run: python main.py")
    else:
        print(f"\\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n\\n👋 Tests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\\n💥 Fatal error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)