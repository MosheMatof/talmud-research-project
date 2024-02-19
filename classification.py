import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import AutoTokenizer, BertModel, BertForMaskedLM
tokenizer = AutoTokenizer.from_pretrained('dicta-il/BEREL_2.0')
model = BertModel.from_pretrained('dicta-il/BEREL_2.0')
# model = BertForMaskedLM.from_pretrained('dicta-il/BEREL_2.0')

def generate_vectors(df, content_column, vectors_file):
    vectors = []
    for _, row in df.iterrows():
        inputs = tokenizer(row[content_column], return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        # Take the mean of the last hidden state to get a single vector that represents the entire text
        mean_vector = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy().flatten().tolist()
        vectors.append(mean_vector)

    # Save vectors as a pickle file
    with open(vectors_file, 'wb') as f:
        pickle.dump(vectors, f)

def classify_vector(df, content_column, label_column):
    df.dropna(subset=[content_column, label_column], inplace=True)

    vectors = []
    labels = []
    for _, row in df.iterrows():
        class_name = row[label_column]
        inputs = tokenizer(row[content_column], return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        cls_token = outputs[0][:, 0, :].detach().numpy().flatten().tolist()
        vectors.append(cls_token)
        labels.append(class_name)

    return vectors, labels

def train_classifier(vectors, labels, Classifier):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42)

    # Train the classifier
    classifier = Classifier()
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

    return classifier

def classify_document(single_document, classifier):
    inputs = tokenizer(single_document, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    single_doc_vector = outputs[0][:, 0, :].detach().numpy().flatten().tolist()

    # Predict the class of the single document
    predicted_class = classifier.predict([single_doc_vector])[0]

    return predicted_class

def compare_with_documents(single_document, class_vectors):
    # Tokenize and encode the single document
    inputs = tokenizer(single_document, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    single_doc_vector = outputs[0][:, 0, :].detach().numpy().flatten().tolist()

    # Calculate cosine similarity with each class of documents
    similarities = {}
    for class_name, vectors in class_vectors.items():
        class_vectors_np = np.array([vec for _, vec in vectors])
        similarity = cosine_similarity([single_doc_vector], class_vectors_np).mean()
        similarities[class_name] = similarity

    # Find the class with the highest average similarity
    best_class = max(similarities, key=similarities.get)

    return best_class