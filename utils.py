import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK datasets
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def preprocess_text(text):
    """
    Comprehensive text preprocessing function
    
    Steps Applied (in order):
    1. Lowercase: Convert all text to lowercase for consistency
    2. Remove Special Characters: Keep only alphanumeric and spaces
    3. Remove Numbers: Remove all digit characters
    4. Tokenization: Split text into individual words
    5. Remove Stopwords: Remove common words (the, is, a, etc.)
    6. Lemmatization: Convert words to base form (running -> run, better -> good)
    """
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Step 1: LOWERCASE all text
    text = text.lower()
    
    # Step 2: REMOVE SPECIAL CHARACTERS
    text = re.sub(r'[^a-z\s\']', '', text)
    
    # Step 3: REMOVE NUMBERS
    text = re.sub(r'\d+', '', text)
    
    # Step 4: TOKENIZATION (split into words)
    tokens = word_tokenize(text)
    
    # Step 5: REMOVE STOPWORDS
    stop_words = set(stopwords.words('english'))
    negations = {"not", "no", "never", "n't"}
    stop_words = stop_words - negations
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # Step 6: LEMMATIZATION
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text
