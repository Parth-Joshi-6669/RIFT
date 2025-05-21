import pandas as pd
import torch
from scipy.spatial.distance import cosine
import open_clip
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import ast
import os

nltk.download('punkt')
nltk.download('wordnet')

# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

# === Load dataset ===
df = pd.read_csv(r"C:\Users\parth\OneDrive\Desktop\Parth\SJSU\Coursework\297\Dataset\Fakeddit\chunks\merged\merged_dataset.csv")

# === Preprocess title ===
lemmatizer = WordNetLemmatizer()

def clean_title(text):
    if pd.isna(text):
        return ""
    tokens = word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    return " ".join(lemmas)

# === Extract text embedding ===
def extract_text_embedding(text):
    if not text.strip():
        return None
    try:
        tokenized = tokenizer([text])
        if isinstance(tokenized, dict):
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
        else:
            tokenized = tokenized.to(device)
        with torch.no_grad():
            features = model.encode_text(tokenized)
        return features.squeeze().cpu().numpy().tolist()
    except Exception as e:
        print(f"❗ Text embedding failed: {e}")
        return None

# === Compute similarity ===
def compute_similarity(emb1, emb2):
    try:
        if emb1 is None or emb2 is None:
            return None
        emb1 = np.asarray(emb1).flatten()
        emb2 = np.asarray(emb2).flatten()
        if emb1.shape != emb2.shape:
            return None
        return 1 - cosine(emb1, emb2)
    except Exception as e:
        print(f"❗ Similarity calc failed: {e}")
        return None

# === Batching and Saving ===
output_dir = r"C:\Users\parth\OneDrive\Desktop\Parth\SJSU\Coursework\297\Dataset\Fakeddit\chunks\merged"
os.makedirs(output_dir, exist_ok=True)

batch_size = 20000
for batch_start in range(220000, len(df), batch_size):
    chunk = df.iloc[batch_start:batch_start + batch_size].copy()

    cleaned_titles = []
    text_embeddings = []
    similarities = []

    for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing batch {batch_start}"):
        clean = clean_title(row.get("title", ""))
        cleaned_titles.append(clean)

        image_emb = row["image_embedding"]
        if isinstance(image_emb, str):
            try:
                image_emb = ast.literal_eval(image_emb)
            except:
                image_emb = None

        text_emb = None
        sim = None
        if image_emb is not None:
            text_emb = extract_text_embedding(clean)
            sim = compute_similarity(text_emb, image_emb)

        text_embeddings.append(text_emb)
        similarities.append(sim)

    chunk["cleaned_title"] = cleaned_titles
    chunk["text_embedding_cleaned"] = text_embeddings
    chunk["embedding_similarity"] = similarities

    output_path = os.path.join(output_dir, f"merged_dataset_final_chunk_{batch_start}.csv")
    chunk.to_csv(output_path, index=False)
    print(f"✅ Saved batch {batch_start} to {output_path}")
