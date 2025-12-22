# Streamlit Sentiment Analysis App - Setup Guide

## Steps to Run the Application:

### Step 1: Run the Notebook

1. Open `main.ipynb` in Jupyter/VS Code
2. Run all cells to train the SVM model
3. The last cells will automatically save:
   - `svm_model.pkl` - Trained SVM model
   - `vectorizer.pkl` - TF-IDF vectorizer

### Step 2: Install Streamlit (if not installed)

```bash
pip install streamlit
```

### Step 3: Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features:

✅ **User-Friendly Interface**: Clean and intuitive UI for entering reviews
✅ **Real-Time Prediction**: Instant sentiment classification
✅ **Text Preprocessing**: Same preprocessing pipeline as training
✅ **Model Information**: Displays model details and confidence
✅ **Responsive Design**: Works on desktop and mobile
✅ **Multiple Sentiment Classes**:

- 😞 Negative (-1)
- 😐 Neutral (0)
- 😊 Positive (1)

## How It Works:

1. User enters a product review
2. Text is preprocessed (same steps as in notebook):
   - Lowercase conversion
   - Special character removal
   - Tokenization
   - Stopword removal
   - Lemmatization
3. TF-IDF vectorizer converts text to numerical features
4. SVM model predicts sentiment class
5. Result is displayed with emoji and color coding

## Files Required:

- `main.ipynb` - Training notebook (run to generate model files)
- `app.py` - Streamlit application
- `svm_model.pkl` - Trained SVM model (auto-generated)
- `vectorizer.pkl` - TF-IDF vectorizer (auto-generated)
- `dataset.csv` - Original training dataset

## Troubleshooting:

**Error: "Model files not found"**
→ Make sure you ran all cells in `main.ipynb` first

**Error: "NLTK data not found"**
→ The app automatically downloads required NLTK data on first run

**App runs slowly**
→ NLTK downloads might be happening in background, wait for completion

## Customization:

You can modify the Streamlit app by editing `app.py`:

- Change colors in CSS section
- Modify sidebar information
- Add more features or visualizations
- Change layout or styling

Enjoy! 🚀
