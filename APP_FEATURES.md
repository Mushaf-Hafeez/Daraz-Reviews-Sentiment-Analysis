# 🔮 Binary Sentiment Analysis App - Features

## Updated Features (December 27, 2025)

### 1. **Binary Classification Only** ✅

- Changed from 3-class to 2-class classification
- **Negative**: -1
- **Positive**: 1
- **Neutral class removed** entirely

### 2. **Single Review Classification** (Tab 1)

- **Input**: Paste or type a single product review
- **Output**: Instant sentiment prediction
- **Shows**:
  - Sentiment label (Negative/Positive)
  - Emoji indicator (😞 / 😊)
  - Class value (-1 / 1)
  - Processed text
  - Model information

### 3. **Batch File Upload & Analysis** (Tab 2) - NEW! 🆕

- **Upload Formats**: PDF, DOC, DOCX
- **Processing**:
  - Extracts all reviews from file
  - Classifies each review individually
  - Calculates sentiment distribution
- **Results Display**:
  - Total reviews analyzed
  - Negative count & percentage
  - Positive count & percentage
  - Distribution visualization
  - Detailed results table (first 10 reviews)
  - Download results as CSV

### 4. **Key Changes Made**

#### File: `app.py`

- ✅ Updated title: "Multi-Class" → "Binary Sentiment Analysis"
- ✅ Removed neutral emoji and class from mappings
- ✅ Added file upload functionality
- ✅ Added batch processing functions:
  - `extract_text_from_pdf()`
  - `extract_text_from_docx()`
  - `extract_text_from_doc()`
  - `parse_reviews_from_text()`
  - `classify_reviews_batch()`
- ✅ Added two tabs: "Single Review" and "Batch Upload"
- ✅ Added percentage calculations and visualization
- ✅ Added CSV download button for results

#### File: `requirements.txt`

**Core ML & NLP Libraries:**

- ✅ `scikit-learn==1.8.0` (SVM, TF-IDF vectorization)
- ✅ `nltk==3.9.2` (Text preprocessing, tokenization, lemmatization)
- ✅ `pandas==2.3.3` (Data manipulation & analysis)
- ✅ `numpy==2.4.0` (Numerical computations)
- ✅ `matplotlib==3.10.8` & `seaborn==0.13.2` (Visualizations)
- ✅ `joblib==1.5.3` (Model persistence)

**File Handling Libraries:**

- ✅ Added: `pypdf==4.0.1` (PDF file handling)
- ✅ Added: `python-docx==0.8.11` (DOCX file handling)

**Web Framework:**

- ✅ `streamlit==1.52.2` (Interactive web app)

### 5. **How to Use**

#### Single Review Classification:

1. Go to "📝 Single Review" tab
2. Paste your review in the text area
3. Click "Classify" button
4. View instant results with sentiment label

#### Batch File Analysis:

1. Go to "📤 Batch Upload" tab
2. Click "Choose a file" and select PDF/DOC/DOCX
3. Review file contains reviews (one per line, minimum 10 characters each)
4. Click "🚀 Analyze All Reviews"
5. View statistics:
   - Total reviews: X
   - Negative: Y (Z%)
   - Positive: A (B%)
6. Download results as CSV

### 6. **Example Output (Batch Upload)**

```
📊 Analysis Results

Total Reviews Analyzed: 1000

😞 Negative Reviews: 200 (20.0%)
😊 Positive Reviews: 800 (80.0%)

📈 Sentiment Distribution
- Negative: 20.0%
- Positive: 80.0%
```

### 7. **Supported File Formats**

- **PDF**: .pdf (using pypdf)
- **Word (2007+)**: .docx (using python-docx)
- **Word (Legacy)**: .doc (basic support using python-docx)

### 8. **Core Technologies Used**

**Machine Learning:**

- SVM (Support Vector Machine) with linear kernel
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- 5000 features extracted (unigrams + bigrams)

**Text Processing:**

- NLTK for tokenization, lemmatization, and stopword removal
- Custom preprocessing with support for code-mixed text (English + Roman Urdu)
- Negation preservation for sentiment context

**Data Science:**

- Pandas for data manipulation
- NumPy for numerical operations
- Matplotlib & Seaborn for visualizations

### 8. **Requirements**

- Python 3.7+
- Streamlit
- PyPDF2 (PDF extraction)
- python-docx (DOCX extraction)
- scikit-learn (pre-trained model)

### 9. **Running the App**

```bash
streamlit run app.py
```

The app will open in your browser at: http://localhost:8501

### 10. **Model Information**

- **Algorithm**: Support Vector Machine (SVM)
- **Vectorizer**: TF-IDF
- **Features**: 5000
- **Accuracy**: ~95.49%
- **Classes**: 2 (Binary)

---

## ✅ All Changes Completed!

Your Streamlit app now supports:

- Single review classification (2 classes)
- Batch file upload and analysis (PDF, DOC, DOCX)
- Percentage-based sentiment distribution
- CSV export of results
- Beautiful UI with visualization
