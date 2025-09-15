-- Customer Feedback Analyzer Database Schema

-- Table to store customer feedback
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    email TEXT,
    product_category TEXT NOT NULL,
    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
    feedback_text TEXT NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to store sentiment analysis results
CREATE TABLE IF NOT EXISTS sentiment_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER,
    sentiment_score REAL NOT NULL,  -- -1 to 1 (negative to positive)
    sentiment_label TEXT NOT NULL,  -- 'positive', 'negative', 'neutral'
    confidence REAL,
    analyzed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (feedback_id) REFERENCES feedback (id)
);

-- Table to store extracted keywords
CREATE TABLE IF NOT EXISTS keywords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feedback_id INTEGER,
    keyword TEXT NOT NULL,
    frequency INTEGER DEFAULT 1,
    relevance_score REAL,
    FOREIGN KEY (feedback_id) REFERENCES feedback (id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_feedback_category ON feedback(product_category);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_date ON feedback(created_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_label ON sentiment_analysis(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_keywords_feedback ON keywords(feedback_id);