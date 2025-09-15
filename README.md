#  Customer Feedback Analyzer  


A Python application for analyzing customer feedback using **Natural Language Processing (NLP)**, **SQLite database**, and **data visualization**.  
Easily extract insights, analyze sentiment, and generate reports from customer reviews.  

---

## ✨ Features
- 🧠 Sentiment Analysis (TextBlob, VADER, custom keywords)  
- 🗄️ SQLite Database for storing and managing feedback  
- 📊 Data Visualization (charts, word clouds, dashboards)  
- 💻 Simple CLI Interface with multiple options  
- 📈 Statistics & Reports for actionable insights  

---

## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/adyasha141/customer-feedback-analyzer.git
cd customer-feedback-analyzer


### 2. Create a virtual environment & install dependencies

python -m venv feedback_env
feedback_env\Scripts\activate         
pip install -r requirements.txt


## 3. Run the app
python main.py



### Example Usage

# Options:
# 1. Add Feedback
# 2. View Feedback
# 3. Analyze Sentiment
# 4. Generate Charts



### Project Structure
customer-feedback-analyzer/
├── main.py              # CLI app
├── demo.py              # Demo script
├── load_sample_data.py  # Load sample entries
├── requirements.txt     # Dependencies
├── database/            # DB operations
├── analysis/            # NLP & sentiment analysis
├── visualization/       # Charts & dashboards
└── data/                # Sample dataset
