import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import time
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from collections import Counter
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_resource_usage():
    process = psutil.Process()
    return psutil.cpu_percent(1), process.memory_info().rss / (1024 ** 2)

def fetch_data():
    with driver.session() as session:
        query = """
        MATCH (p:Post)
        OPTIONAL MATCH (p)-[:POSTED_BY]->(u:User)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(s:Subreddit)
        RETURN p.post_id AS post_id, 
               COALESCE(p.upvotes, 0) AS upvotes, 
               COALESCE(p.label, 0) AS label, 
               COALESCE(p.embedding_similarity, 0.0) AS embedding_similarity,
               COALESCE(u.user_id, "Unknown") AS author, 
               COALESCE(s.name, "Unknown") AS subreddit
        """
        return pd.DataFrame([dict(r) for r in session.run(query)])

df = fetch_data()
label_encoders = {"author": LabelEncoder(), "subreddit": LabelEncoder()}
for col in ["author", "subreddit"]:
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

post_ids = df["post_id"].astype(str).tolist()
author_ids = df["author"].astype(str).tolist()
all_nodes = sorted(set(post_ids + author_ids))
node_to_index = {node: i for i, node in enumerate(all_nodes)}
num_nodes = len(all_nodes)

# Feature matrix
X_np = np.zeros((num_nodes, 4))
for _, row in df.iterrows():
    idx = node_to_index[str(row["post_id"])]
    X_np[idx] = [
        row["upvotes"],
        row["author"],
        row["subreddit"],
        row["embedding_similarity"]
    ]
X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
X = torch.tensor(X_np, dtype=torch.float)

# Labels
y_full = torch.full((num_nodes,), -1, dtype=torch.long)
for _, row in df.iterrows():
    idx = node_to_index[str(row["post_id"])]
    y_full[idx] = int(row["label"])
mask = y_full != -1
print("📊 Label distribution:", Counter(y_full[mask].cpu().numpy()))

train_idx, test_idx = train_test_split(mask.nonzero(as_tuple=True)[0], test_size=0.2, random_state=42)

def fetch_edges():
    with driver.session() as session:
        query = """
        MATCH (u:User)-[:POSTED_BY]->(p:Post)
        RETURN u.user_id AS source, p.post_id AS target
        """
        return [(str(r["source"]), str(r["target"])) for r in session.run(query)]

edges_raw = fetch_edges()
edges = [(node_to_index[src], node_to_index[tgt]) for src, tgt in edges_raw if src in node_to_index and tgt in node_to_index]
edges += [(tgt, src) for (src, tgt) in edges]  # bidirectional
edges += [(i, i) for i in range(num_nodes)]    # self-loops
edge_index = torch.tensor(edges, dtype=torch.long).T

# GraphSAGE Model with BatchNorm and 4 layers
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.conv4 = SAGEConv(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        return F.log_softmax(self.conv4(x, edge_index), dim=1)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(in_dim=4, hidden=128, out_dim=2).to(device)
X, y = X.to(device), y_full.to(device)
edge_index = edge_index.to(device)
train_idx, test_idx = train_idx.to(device), test_idx.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index)
    loss = F.nll_loss(out[train_idx], y[train_idx])
    if torch.isnan(loss):
        print("❌ NaN detected in loss — stopping.")
        return float('nan')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
    optimizer.step()
    return loss.item()

def evaluate_model():
    model.eval()
    out = model(X, edge_index)
    probs = torch.exp(out[test_idx]).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    true = y[test_idx].cpu().numpy()
    roc = roc_auc_score(true, probs[:, 1])
    f1 = f1_score(true, preds, average="weighted")
    precision = precision_score(true, preds, average="weighted")
    recall = recall_score(true, preds, average="weighted")
    print(f"📊 ROC-AUC: {roc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Training loop
start = time.time()
for epoch in range(100):
    loss = train()
    if torch.isnan(torch.tensor(loss)):
        break
    scheduler.step(loss)
    if epoch % 30 == 0:
        acc = (model(X, edge_index).argmax(1)[test_idx] == y[test_idx]).float().mean().item()
        cpu, ram = get_resource_usage()
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.4f}, CPU: {cpu:.2f}%, RAM: {ram:.2f} MB")

# Final stats
elapsed = time.time() - start
final_acc = (model(X, edge_index).argmax(1)[test_idx] == y[test_idx]).float().mean().item()
cpu, ram = get_resource_usage()
print("\n✅ Training complete!")
print(f"🕒 Time: {elapsed:.2f}s")
print(f"🎯 Accuracy: {final_acc * 100:.2f}%")
evaluate_model()
print(f"💻 CPU: {cpu:.2f}%, 🖥️ RAM: {ram:.2f} MB")
if torch.cuda.is_available():
    print(f"⚡ GPU Mem: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
