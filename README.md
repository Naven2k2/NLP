# Hate Speech Classification using Simple RNN

## 📌 Problem Statement

The objective of this project is to build a binary classifier to detect hate speech from tweets using a Simple Recurrent Neural Network (RNN).

---

## 📂 Dataset

* File: `imbalanced_data.csv`
* Columns used:

  * `tweet` → text data
  * `label` → target (0 = non-hate, 1 = hate)

### ⚠️ Dataset Imbalance

* Non-hate: ~29,000
* Hate: ~2,200
  The dataset is highly imbalanced, which affects model performance.

---

## 🧹 Data Preprocessing

The following steps were applied:

* Converted text to lowercase
* Removed:

  * @mentions
  * URLs
  * Special characters
  * Numbers
* Removed extra spaces

---

## 🔤 Tokenization & Vocabulary

* Tokenized tweets into words
* Selected top **5000 most frequent words**
* Added special tokens:

  * `<PAD>` = 0
  * `<OOV>` = 1
* Final vocabulary size: ~5002

---

## 🔢 Text Representation

* Converted text into integer sequences
* Applied padding:

  * Maximum sequence length = 25

---

## 🔀 Train-Test Split

* 80% training / 20% testing
* Used **stratified sampling**

---

## 🧠 Model Architecture

* Embedding Layer (64 dimensions)
* SimpleRNN Layer (64 units)
* Dense Output Layer (Sigmoid activation)

---

## ⚙️ Training Details

* Loss Function: Binary Crossentropy
* Optimizer: Adam
* Epochs: 10
* Batch Size: 32
* Validation Split: 20%

---

## 📊 Evaluation Metrics

* Accuracy
* F1 Score
* Confusion Matrix

---

## 📈 Results

* Accuracy: ~93%
* F1 Score: ~0.55

### Confusion Matrix:

[[5694  251]
[ 183  265]]

---

## ⚠️ Key Observation

Although accuracy is high, the F1 score is moderate due to class imbalance. The model performs better on the majority class (non-hate) than the minority class (hate speech).

---

## 🚀 Improvements Applied

* Threshold tuning to improve F1 score

---

## 🔥 Future Improvements

* Apply class weights to handle imbalance
* Add Dropout to reduce overfitting
* Use Early Stopping for better generalization
* Improve F1 score further

---

## 📁 Project Structure

project/
│
├── hate_speech_rnn.py
├── imbalanced_data.csv
├── README.md

---

## ▶️ How to Run

1. Install dependencies:
   pip install pandas numpy scikit-learn tensorflow

2. Run the script:
   python hate_speech_rnn.py

---

## 🧠 Conclusion

This project demonstrates how a Simple RNN can be applied to text classification. Handling class imbalance is crucial for improving real-world performance.

---

## 💡 Author

Mothe Naveen
