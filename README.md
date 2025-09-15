# 🎯 Customer Feedback Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NLP](https://img.shields.io/badge/NLP-TextBlob%20%7C%20VADER-green.svg)](https://github.com/sloria/TextBlob)

A comprehensive Python application for analyzing customer feedback using **Natural Language Processing (NLP)**, **SQLite database**, and **data visualization**. Perfect for learning NLP concepts, database operations, and Python project structure.

**Author:** Adyasha Khandai  
**Copyright:** © 2025 Adyasha Khandai  
**License:** MIT License

## ✨ Features

- 🗄️ **SQLite Database**: Store and manage customer feedback with proper schema design
- 🧠 **Sentiment Analysis**: Multi-method approach using TextBlob, VADER, and custom keywords
- 🔍 **Text Processing**: Advanced preprocessing with tokenization, lemmatization, and stopword removal
- 📊 **Data Visualization**: Generate charts, word clouds, and interactive dashboards
- 💻 **CLI Interface**: User-friendly command-line application with 8+ features
- 🔄 **Batch Processing**: Analyze multiple feedback entries simultaneously
- 📈 **Statistics & Reports**: Comprehensive analytics and insights
- 🎯 **Search & Filter**: Advanced filtering by category, rating, and sentiment

## 🎓 Learning Objectives

- **Python Programming**: Object-oriented design, error handling, modular architecture
- **Natural Language Processing**: Sentiment analysis, text preprocessing, keyword extraction
- **Database Operations**: SQLite schema design, queries, relationships, and indexes
- **Data Visualization**: Creating meaningful charts and dashboards with matplotlib/seaborn
- **Software Engineering**: Project structure, testing, documentation, and deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdyashaK/customer-feedback-analyzer.git
   cd customer-feedback-analyzer
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv feedback_env
   feedback_env\Scripts\activate
   
   # Linux/Mac
   python -m venv feedback_env
   source feedback_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Load sample data (recommended):**
   ```bash
   python load_sample_data.py
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

### Alternative: Quick Demo
To see all features without interaction:
```bash
python demo.py
```

## 📁 Project Structure

```
customer-feedback-analyzer/
├── 📄 main.py                    # Interactive CLI application
├── 🎯 demo.py                    # Non-interactive demo
├── 📊 load_sample_data.py        # Sample data loader
├── 🧪 test_system.py             # System tests
├── 📋 requirements.txt           # Python dependencies
├── 📜 LICENSE                    # MIT License
├── 📖 README.md                  # Project documentation
├── 🗄️ database/
│   ├── __init__.py
│   ├── db_manager.py             # Database operations & CRUD
│   └── schema.sql                # SQLite database schema
├── 🧠 analysis/
│   ├── __init__.py
│   ├── sentiment_analyzer.py     # Multi-method sentiment analysis
│   └── text_preprocessor.py      # Text cleaning & NLP preprocessing
├── 📊 visualization/
│   ├── __init__.py
│   └── charts.py                 # Charts, graphs & dashboards
└── 📂 data/
    └── sample_feedback.csv       # 20 realistic feedback samples
```

## 🎮 Usage Examples

### Interactive CLI Application
```bash
python main.py

# Menu Options:
# 1. 📝 Add New Feedback
# 2. 🔍 View All Feedback  
# 3. 🧠 Analyze Sentiment (All Feedback)
# 4. 📈 Generate Visualizations
# 5. 📊 View Statistics & Reports
# 6. 🔍 Search & Filter Feedback
# 7. ⚙️ Manage Data
# 8. ❓ Help & Information
```

### Quick Demo (No Interaction Required)
```bash
python demo.py
# Shows all features, generates charts, displays statistics
```

### Load Sample Data
```bash
python load_sample_data.py
# Loads 20 feedback entries and runs sentiment analysis
```

### Run System Tests
```bash
python test_system.py
# Tests all components: database, NLP, visualization
```

## 🔧 API Usage

### Sentiment Analysis
```python
from analysis.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_sentiment_comprehensive(
    "This product is absolutely amazing!"
)

print(result['overall']['overall_label'])  # 'positive'
print(result['overall']['overall_score'])  # 0.75
```

### Database Operations
```python
from database.db_manager import DatabaseManager

db = DatabaseManager()

# Add feedback
feedback_id = db.add_feedback(
    customer_name="John Doe",
    email="john@example.com",
    product_category="Electronics",
    rating=5,
    feedback_text="Great product, highly recommended!"
)

# Get statistics
stats = db.get_database_stats()
print(f"Total feedback: {stats['total_feedback']}")
```

### Visualization
```python
from visualization.charts import FeedbackVisualizer
from database.db_manager import DatabaseManager

db = DatabaseManager()
visualizer = FeedbackVisualizer()

# Get data and create charts
feedback_data = db.get_feedback_with_sentiment()
visualizer.plot_sentiment_distribution(feedback_data)
visualizer.create_summary_dashboard(feedback_data)
```

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all system tests
python test_system.py

# Test individual components
python -c "from analysis.sentiment_analyzer import SentimentAnalyzer; print('Sentiment OK')"
python -c "from database.db_manager import DatabaseManager; print('Database OK')"
```

## 📊 Sample Data

The project includes 20 realistic customer feedback samples covering:
- **8 Product Categories**: Electronics, Clothing, Food & Beverage, Healthcare, Software, Books, Home & Garden, Sports
- **Rating Distribution**: 1-5 stars with realistic distribution
- **Sentiment Variety**: Positive (70%), Negative (30%)
- **Text Complexity**: Various lengths and writing styles

## 🎯 NLP Techniques Used

### Text Preprocessing
- **Tokenization**: Breaking text into individual words/tokens
- **Stop Word Removal**: Filtering common words (the, and, or, etc.)
- **Lemmatization**: Converting words to their root form
- **Text Cleaning**: URL removal, normalization, whitespace handling

### Sentiment Analysis Methods
1. **TextBlob**: Rule-based sentiment analysis with polarity scoring
2. **VADER**: Lexicon and rule-based tool optimized for social media text
3. **Custom Keywords**: Domain-specific positive/negative word matching
4. **Ensemble Approach**: Combining multiple methods for better accuracy

### Feature Extraction
- **Keyword Extraction**: TF-IDF style relevance scoring
- **N-grams**: Bigrams and trigrams for phrase detection
- **Emotion Detection**: Basic emotion classification

## 📈 Generated Visualizations

The system generates several types of charts:

1. **Sentiment Distribution** - Pie chart showing positive/negative/neutral breakdown
2. **Rating Distribution** - Bar chart of 1-5 star ratings
3. **Category Analysis** - Horizontal bar chart of feedback by product category
4. **Word Cloud** - Visual representation of most frequent words
5. **Summary Dashboard** - Combined view with multiple charts
6. **Trend Analysis** - Sentiment changes over time (if date data available)

## 🛠️ Troubleshooting

### Common Issues

**NLTK Data Missing:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

**Virtual Environment Issues:**
```bash
# Windows
deactivate
rmdir /s feedback_env
python -m venv feedback_env
feedback_env\Scripts\activate
pip install -r requirements.txt
```

**Database Issues:**
```bash
# Reset database
del feedback_analyzer.db
python load_sample_data.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Learning Resources

- [Natural Language Processing with Python](https://www.nltk.org/book/)
- [TextBlob Documentation](https://textblob.readthedocs.io/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [SQLite Tutorial](https://www.sqlite.org/docs.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Adyasha Khandai**
- GitHub: [@AdyashaK](https://github.com/AdyashaK)
- Project: Customer Feedback Analyzer
- Year: 2025

---

⭐ **If you found this project helpful for learning NLP and Python, please give it a star!** ⭐
#   C u s t o m e r   F e e d b a c k   A n a l y z e r  
 #   C u s t o m e r   F e e d b a c k   A n a l y z e r  
 