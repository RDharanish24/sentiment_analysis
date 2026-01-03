# IMDb Movie Review Sentiment Analysis using LSTM

This project performs **sentiment analysis** on IMDb movie reviews using a **Deep Learning LSTM model** built with **TensorFlow/Keras**.  
The model classifies reviews as **positive** or **negative**.

---

## 📌 Dataset
- **IMDb Dataset of 50K Movie Reviews**
- Source: Kaggle  
- Contains 50,000 labeled movie reviews (positive / negative)

---

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Kaggle API

---

## ⚙️ Project Workflow
1. Download IMDb dataset using Kaggle API  
2. Load and preprocess data  
3. Convert sentiment labels to numeric values  
4. Tokenize and pad text sequences  
5. Build an LSTM-based neural network  
6. Train the model  
7. Evaluate model performance  
8. Predict sentiment for custom reviews  

---

## 🧠 Model Architecture
- Embedding Layer (5000 words, 128 dimensions)
- LSTM Layer (128 units)
- Dense Output Layer (Sigmoid activation)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/RDharanish24/sentiment_analysis.git
cd sentiment_analysis
```

### 2️⃣ Install Required Libraries
```
pip install -r requirements.txt 
```
### 3️⃣ Setup Kaggle API
```
Download kaggle.json from Kaggle Account Settings

Place it in the project directory
```
### 4️⃣ Run the Script
```
python sentiment_analysis.py
```


## 📊 Model Performance

Loss and Accuracy are printed after evaluation on test data

Trained for 5 epochs

Uses 80/20 train-test split

✍️ Example Prediction
review = "i do not like this movie"
sentiment = predict_sentiment(review)
print(sentiment)


Output:

negative

### 📁 Files in the Repository
├── sentiment_analysis.py
├── README.md
├──requiremnts.txt

### 📌 Notes

Ensure Kaggle API credentials are valid

GPU recommended for faster training (optional)
