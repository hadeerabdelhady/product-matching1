import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Load Data
master_df = pd.read_excel("Product Matching Dataset.xlsx", sheet_name="Master File")
dataset_df = pd.read_excel("Product Matching Dataset.xlsx", sheet_name="Dataset")

# Arabic Text Cleaning Function
def clean_arabic_text(text):
    if pd.isna(text):# if there is missing value replace it by ""
        return ""
    text = str(text).strip()# remove spaces 
    
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces between words and each other
    text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove diacritics (التشكيل على الحروف)
    return text

# Apply Text Cleaning to Arabic Names
master_df["clean_product_name_ar"] = master_df["product_name_ar"].apply(clean_arabic_text)
dataset_df["clean_seller_name_ar"] = dataset_df["marketplace_product_name_ar"].apply(clean_arabic_text)

# TF-IDF Vectorization char-wb work on charcter level instead of word level helping in spelling mistakes
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
tfidf_matrix = tfidf.fit_transform(pd.concat([master_df["clean_product_name_ar"], dataset_df["clean_seller_name_ar"]]))

# Compute Cosine Similarity , Output is a similarity matrix, where each row represents a master product, and each column represents a seller product.
cosine_sim = cosine_similarity(tfidf_matrix[:len(master_df)], tfidf_matrix[len(master_df):])

# Assign SKU Based on Highest Similarity
dataset_df["matched_sku"] = None
dataset_df["similarity_score"] = 0.0
times = [] 
for i in range(len(dataset_df)):
    start_time = time.time()  # Start timing
    best_match_idx = np.argmax(cosine_sim[:, i])  # Index of highest similarity
    best_match_score = np.max(cosine_sim[:, i])  # Highest similarity score
    end_time = time.time()  # End timing
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    times.append(execution_time)
    
    if best_match_score >= 0.85:  # Only assign if confidence is high
        dataset_df.at[i, "matched_sku"] = master_df.iloc[best_match_idx]["sku"]
        dataset_df.at[i, "similarity_score"] = best_match_score

# Calculate average processing time per record
avg_time = np.mean(times)
max_time = np.max(times)

print(f"Average Matching Time: {avg_time:.2f} ms")
print(f"Max Matching Time: {max_time:.2f} ms")

# Split Data
X = dataset_df[["similarity_score"]].values # # Feature: Similarity score
y = np.array([1 if sku is not None else 0 for sku in dataset_df["matched_sku"]])  # Target: 1 (matched), 0 (not matched)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
dataset_df["confidence"] = np.where(dataset_df["similarity_score"] >= 0.85, "High", "Low")


# Check if the assigned SKU matches the correct SKU
dataset_df["correct_match"] = dataset_df["matched_sku"] == dataset_df["sku"]

new_df=dataset_df
new_df.to_excel('new_df.xlsx',index=False)




# Calculate Accuracy
accuracy = dataset_df["correct_match"].mean() * 100  # Convert to percentage
print(f"Matching Accuracy: {accuracy:.2f}%")