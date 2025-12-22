import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

################################################################################
# Code to Solve the Product Error.
################################################################################

import nltk
import os

# Set custom NLTK data path (Streamlit Cloud pe yeh writable hai)
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Add this path to NLTK's search paths
nltk.data.path.append(nltk_data_dir)

# Download required resources if not found (quiet=True to avoid extra output)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)  # punkt_tab ke liye punkt bhi chahiye

# Agar aap stopwords ya wordnet bhi use kar rahe ho (lemmatizer ke liye)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)

################################################################################

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
