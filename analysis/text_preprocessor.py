"""
Text Preprocessing Module for Customer Feedback Analyzer

This module handles text cleaning and preprocessing tasks:
- Text normalization and cleaning
- Stop word removal
- Tokenization
- Lemmatization/Stemming
- Special character handling

Author: Adyasha Khandai
Copyright (c) 2025 Adyasha Khandai
License: MIT License
"""

import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Download required NLTK data (only needs to be done once)
def download_nltk_data():
    """Download necessary NLTK data files."""
    nltk_downloads = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'vader_lexicon'
    ]
    
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {item}: {e}")


class TextPreprocessor:
    """Handles all text preprocessing operations."""
    
    def __init__(self, language='english'):
        """
        Initialize the text preprocessor.
        
        Args:
            language: Language for stop words and other language-specific operations
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        
        # Download NLTK data if not already available
        download_nltk_data()
        
        # Load stop words
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            print(f"Warning: Could not load stopwords for {language}, using English")
            self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words for feedback analysis
        custom_stops = {
            'product', 'service', 'company', 'business', 'customer',
            'would', 'could', 'should', 'really', 'much', 'well',
            'good', 'bad', 'ok', 'okay', 'fine', 'nice', 'great'
        }
        self.stop_words.update(custom_stops)
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_punctuation(self, text: str, keep_sentences=False) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text: Text to process
            keep_sentences: If True, keep sentence-ending punctuation
            
        Returns:
            Text with punctuation removed
        """
        if not text:
            return ""
        
        if keep_sentences:
            # Keep sentence endings for better analysis
            text = re.sub(r'[^\\w\\s.!?]', '', text)
        else:
            # Remove all punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text
    
    def tokenize(self, text: str, sentences=False) -> List[str]:
        """
        Tokenize text into words or sentences.
        
        Args:
            text: Text to tokenize
            sentences: If True, tokenize into sentences; otherwise into words
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            if sentences:
                return sent_tokenize(text)
            else:
                return word_tokenize(text)
        except Exception as e:
            print(f"Warning: Tokenization failed: {e}")
            # Fallback to simple splitting
            if sentences:
                return text.split('.')
            else:
                return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens with stop words removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their root form.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        lemmatized = []
        
        # Get POS tags for better lemmatization
        pos_tags = pos_tag(tokens)
        
        for token, pos in pos_tags:
            # Convert POS tag to format expected by WordNetLemmatizer
            pos_tag_simplified = self._get_wordnet_pos(pos)
            lemma = self.lemmatizer.lemmatize(token.lower(), pos_tag_simplified)
            lemmatized.append(lemma)
        
        return lemmatized
    
    def _get_wordnet_pos(self, pos_tag: str) -> str:
        """
        Convert NLTK POS tag to WordNet POS tag.
        
        Args:
            pos_tag: NLTK POS tag
            
        Returns:
            WordNet POS tag
        """
        if pos_tag.startswith('J'):
            return 'a'  # adjective
        elif pos_tag.startswith('V'):
            return 'v'  # verb
        elif pos_tag.startswith('N'):
            return 'n'  # noun
        elif pos_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
    
    def filter_tokens(self, tokens: List[str], min_length: int = 2, max_length: int = 20) -> List[str]:
        """
        Filter tokens by length and content.
        
        Args:
            tokens: List of tokens to filter
            min_length: Minimum token length
            max_length: Maximum token length
            
        Returns:
            Filtered list of tokens
        """
        filtered = []
        
        for token in tokens:
            # Skip if too short or too long
            if not (min_length <= len(token) <= max_length):
                continue
            
            # Skip if not alphabetic (numbers, special chars)
            if not token.isalpha():
                continue
            
            # Skip if all uppercase (likely acronym or shouting)
            if token.isupper() and len(token) > 2:
                continue
            
            filtered.append(token)
        
        return filtered
    
    def preprocess_for_sentiment(self, text: str) -> str:
        """
        Preprocess text specifically for sentiment analysis.
        Preserves emoticons and sentiment-bearing punctuation.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text suitable for sentiment analysis
        """
        if not text:
            return ""
        
        # Convert to lowercase but preserve emoticons
        # Simple emoticon preservation
        emoticons = re.findall(r'[:;=8][-~]?[)\](<>]|[\](<>:;=8][-~]?[:;=8]', text)
        
        # Clean text
        processed = self.clean_text(text)
        
        # Add emoticons back
        if emoticons:
            processed += ' ' + ' '.join(emoticons)
        
        return processed
    
    def preprocess_for_keywords(self, text: str) -> List[str]:
        """
        Preprocess text specifically for keyword extraction.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            List of processed tokens suitable for keyword extraction
        """
        if not text:
            return []
        
        # Clean and tokenize
        cleaned = self.clean_text(text)
        cleaned = self.remove_punctuation(cleaned)
        tokens = self.tokenize(cleaned)
        
        # Remove stop words
        tokens = self.remove_stopwords(tokens)
        
        # Filter tokens
        tokens = self.filter_tokens(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        return tokens
    
    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """
        Generate n-grams from tokens.
        
        Args:
            tokens: List of tokens
            n: Size of n-grams
            
        Returns:
            List of n-gram strings
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """
        Extract key phrases using simple statistical methods.
        
        Args:
            text: Text to analyze
            max_phrases: Maximum number of phrases to return
            
        Returns:
            List of key phrases
        """
        # Get preprocessed tokens
        tokens = self.preprocess_for_keywords(text)
        
        if len(tokens) < 2:
            return tokens
        
        # Get bigrams and trigrams
        bigrams = self.get_ngrams(tokens, 2)
        trigrams = self.get_ngrams(tokens, 3)
        
        # Combine all candidates
        candidates = tokens + bigrams + trigrams
        
        # Simple frequency-based selection
        from collections import Counter
        phrase_counts = Counter(candidates)
        
        # Get top phrases
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(max_phrases)]
        
        return top_phrases


# Example usage and testing
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # Test text
    sample_text = """
    This product is AMAZING! I absolutely love it. The quality is outstanding and the 
    customer service was excellent. However, the delivery took a bit longer than expected.
    Overall, I would definitely recommend this to others. Great job! :)
    """
    
    print("Original text:", sample_text)
    print("\\nCleaned text:", preprocessor.clean_text(sample_text))
    print("\\nTokens:", preprocessor.tokenize(preprocessor.clean_text(sample_text)))
    print("\\nProcessed for sentiment:", preprocessor.preprocess_for_sentiment(sample_text))
    print("\\nKeyword tokens:", preprocessor.preprocess_for_keywords(sample_text))
    print("\\nKey phrases:", preprocessor.extract_key_phrases(sample_text))