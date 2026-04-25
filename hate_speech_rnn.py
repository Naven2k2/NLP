import pandas as pd
import numpy as np
import re
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("imbalanced_data.csv")
df = df[['tweet', 'label']]

print("Class distribution:\n", df['label'].value_counts())


# -------------------------------
# 2. TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_tweet'] = df['tweet'].apply(clean_text)


# -------------------------------
# 3. TOKENIZATION & VOCAB
# -------------------------------
tokenized = df['clean_tweet'].apply(lambda x: x.split())

all_words = [word for tokens in tokenized for word in tokens]
word_counts = Counter(all_words)

vocab = word_counts.most_common(5000)

word2idx = {word: idx+2 for idx, (word, _) in enumerate(vocab)}

# Special tokens
word2idx["<PAD>"] = 0
word2idx["<OOV>"] = 1

vocab_size = len(word2idx)

print("Vocab size:", vocab_size)


# -------------------------------
# 4. TEXT → SEQUENCES
# -------------------------------
def text_to_sequence(tokens):
    return [word2idx.get(word, word2idx["<OOV>"]) for word in tokens]

sequences = tokenized.apply(text_to_sequence)

max_len = 25
X = pad_sequences(sequences, maxlen=max_len, padding='post')

y = df['label'].values


# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# -------------------------------
# 6. MODEL BUILDING
# -------------------------------
model = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=vocab_size, output_dim=64),
    SimpleRNN(64),
    Dense(1, activation='sigmoid')
])

model.summary()


# -------------------------------
# 7. TRAINING
# -------------------------------
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)


# -------------------------------
# 8. EVALUATION
# -------------------------------
y_pred = (model.predict(X_test) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nTest Accuracy:", accuracy)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)