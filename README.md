# sms-spam-classifier

````markdown
# SMS Spam Classifier

A comprehensive project aimed at building an SMS spam detection system using machine learning. Key steps include data cleaning, exploratory analysis, preprocessing, model building, evaluation, improvement, a user-facing website, and deployment of the best-performing model (Multinomial Naive Bayes).


## Prerequisites

Make sure you have:

- Python 3.7 or higher  
- Key dependencies (install with):

  ```bash
  pip install -r requirements.txt
````

**Typical packages include:** `pandas`, `numpy`, `scikit-learn`, `nltk` (or `spaCy`), `flask`, `joblib` (or `pickle`), and visualization tools like `matplotlib`, `seaborn`, or `plotly`.

---

## Getting Started

### 1. Data Cleaning

* Load the raw SMS dataset (e.g., UCI SMS Spam Collection).
* Remove unnecessary columns, handle missing values, and standardize formatting.
* Save the cleaned data for further use.

### 2. Exploratory Data Analysis (EDA)

* Analyze class distribution: spam vs. ham.
* Visualize message length distributions.
* Generate word clouds or frequency plots for common terms in each class.
* Identify keywords or patterns typical of spam vs. ham.

### 3. Text Preprocessing

* Tokenization, lowercasing.
* Removal of punctuation, numbers, stop words, and extra whitespace.
* (Optionally) Lemmatization or stemming.
* Transform text into numerical features using methods like:

  * Count Vectorizer (Bag of Words)
  * TF-IDF
* Save the fitted vectorizer for training and deployment.

### 4. Model Building

* Train a Multinomial Naive Bayes classifier using the vectorized text features and labels.
* Optionally, explore other classifiers (SVM, Logistic Regression, Random Forest, etc.) for baseline comparison.

### 5. Model Evaluation

* Split the dataset into training and test sets.
* Evaluate using metrics such as:

  * Accuracy
  * Precision, Recall
  * F1-Score
  * Confusion Matrix (if desired)
* Compare performance to select the best model.

### 6. Improvement Strategies

Consider:

* Hyperparameter tuning (e.g., alpha smoothing for Naive Bayes).
* Experimenting with n-grams (bi-grams, tri-grams).
* Advanced features: count of special characters, uppercase ratios, spammy keywords, etc.
* Trying TF-IDF instead of raw counts.

### 7. Web Application

* Build a simple app that:

  1. Accepts SMS input.
  2. Applies preprocessing and vectorization.
  3. Predicts using the trained Multinomial NB.
  4. Returns a user-friendly spam/ham result.


### 8. Deployment of Best Algorithm

* Save the best model (`multinomial_nb.pkl`) and vectorizer (`vectorizer.pkl`).
* Load them in the Flask application for real-time inference.
* Host the app locally or deploy to platforms like Heroku, AWS, or Render.

---

## Results

Provide your key findings here:

* **Multinomial Naive Bayes performance**:

  * Accuracy: *e.g.,* 98%
  * Precision (Spam): *e.g.,* 95%
  * Recall (Spam): *e.g.,* 90%
  * F1-Score: *e.g.,* 92%


---

## License

This project is licensed under the MIT License — see the [`LICENSE`](LICENSE) file for details.

---


* **Author**: Ayushi Tyagi
* **GitHub**: [ayushi-tyagi080](https://github.com/ayushi-tyagi080)


