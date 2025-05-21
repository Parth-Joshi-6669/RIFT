import praw
import pandas as pd
import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
import tqdm
import time
from PIL import Image
import torch
from io import BytesIO
import open_clip
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT, DATASET_FOLDER
from scipy.spatial.distance import cosine

# Initialize PRAW
reddit = praw.Reddit(client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

# Saving some information for faster fetch
subreddit_meta_cache = {}

# CLIP model setup with GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()

# Optimization Settings
MAX_WORKERS = 4
CHUNK_SIZE = 5000
REQUEST_TIMEOUT = 5
SLEEP_AFTER_CHUNK = 1  # seconds

# Set output folder for chunks
CHUNK_FOLDER = os.path.join(DATASET_FOLDER, "chunks")
os.makedirs(CHUNK_FOLDER, exist_ok=True)

def extract_image_embedding(url):
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(image_tensor)
        return features.squeeze().cpu().numpy().tolist()
    except:
        return None

def extract_text_embedding(text):
    try:
        text_inputs = tokenizer([text], truncate=True)
        with torch.no_grad():
            features = clip_model.encode_text(text_inputs)
        return features.squeeze().cpu().numpy().tolist()
    except:
        return None

def compute_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return None
    return 1 - cosine(embedding1, embedding2)

def is_valid_url(url):
    try:
        response = requests.head(url, timeout=REQUEST_TIMEOUT)
        return response.status_code == 200
    except:
        return False

def fetch_post_data(row, extracted_data, missing_posts):
    post_id = row["id"]
    fallback = False

    if not isinstance(row.get("title"), str) or pd.isna(row.get("title")):
        fallback = True
    if "image_url" in row and row["image_url"] and not is_valid_url(row["image_url"]):
        fallback = True

    image_embedding = None
    if "image_url" in row and row["image_url"] and not fallback:
        image_embedding = extract_image_embedding(row["image_url"])

    text_embedding = None
    similarity_score = None

    title = str(row.get("title", "") if pd.notna(row.get("title")) else "")
    text = str(row.get("text", "") if pd.notna(row.get("text")) else "")
    full_text = title + " " + text

    if full_text.strip():
        text_embedding = extract_text_embedding(full_text)

    if image_embedding and text_embedding:
        similarity_score = compute_similarity(image_embedding, text_embedding)

    try:
        post = reddit.submission(id=post_id)
        post_data = {
            "post_id": post.id,
            "title": post.title,
            "selftext": post.selftext if post.selftext else "",
            "num_comments": post.num_comments,
            "upvotes": post.score,
            "created_utc": post.created_utc,
            "author_id": str(post.author) if post.author else None,
            "subreddit": post.subreddit.display_name,
            "url": post.url if hasattr(post, "url") else "",
            "label": row.get("2_way_label", None),
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "image_text_similarity": similarity_score
        }
    except Exception as e:
        missing_posts.append(post_id)
        post_data = {
            "post_id": post_id,
            "title": row.get("title", ""),
            "selftext": row.get("text", ""),
            "num_comments": row.get("num_comments", 0),
            "upvotes": row.get("upvotes", 0),
            "created_utc": row.get("created_utc", ""),
            "author_id": row.get("author", None),
            "subreddit": row.get("subreddit", ""),
            "url": row.get("image_url", ""),
            "label": row.get("2_way_label", None),
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "image_text_similarity": similarity_score
        }

    extracted_data.append(post_data)

def process_chunk(df_chunk, chunk_index, original_file):
    extracted_data = []
    missing_posts = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm.tqdm(executor.map(lambda row: fetch_post_data(row, extracted_data, missing_posts), df_chunk.to_dict("records")), total=len(df_chunk)))

    # Save extracted chunk
    chunk_output_path = os.path.join(CHUNK_FOLDER, f"extracted_chunk_{original_file}_part_{chunk_index}.csv")
    pd.DataFrame(extracted_data).to_csv(chunk_output_path, index=False)

    # Save missing posts for this chunk
    missing_output_path = os.path.join(CHUNK_FOLDER, f"missing_posts_chunk_{original_file}_part_{chunk_index}.json")
    with open(missing_output_path, "w") as f:
        json.dump(missing_posts, f)

    print(f"✅ Finished chunk {chunk_index}. Sleeping for {SLEEP_AFTER_CHUNK} seconds to cool down.")
    time.sleep(SLEEP_AFTER_CHUNK)

# List of TSV files to process
tsv_files = ["all_validate.tsv", "all_test_public.tsv"]

for file in tsv_files:
    file_path = os.path.join(DATASET_FOLDER, file)
    df = pd.read_csv(file_path, sep="\t")
    total_chunks = (len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(total_chunks):
        chunk_output_path = os.path.join(CHUNK_FOLDER, f"extracted_chunk_{file}_part_{i}.csv")
        if os.path.exists(chunk_output_path):
            print(f"Skipping already processed chunk: {chunk_output_path}")
            continue
        print(f"Processing chunk {i+1}/{total_chunks} of file {file}")
        chunk = df.iloc[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        process_chunk(chunk, i, file)

print("Data extraction complete. All chunks processed.")
