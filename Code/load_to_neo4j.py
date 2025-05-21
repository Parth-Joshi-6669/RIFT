from neo4j import GraphDatabase
import pandas as pd
import os
from config import DATASET_FOLDER, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Establish connection to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Cypher query to create post nodes
def create_post_node(tx, post_data):
    query = (
        "MERGE (p:Post {post_id: $post_id}) "
        "SET p.title = $title, "
        "p.upvotes = $upvotes, p.created_utc = $created_utc, p.url = $url, p.label = $label, "
        "p.embedding_similarity = $embedding_similarity"
    )
    tx.run(query, **post_data)

# Cypher query to create user nodes
def create_user_node(tx, user_data):
    query = "MERGE (u:User {user_id: $user_id})"
    tx.run(query, **user_data)

# Cypher query to create subreddit nodes
def create_subreddit_node(tx, subreddit_name):
    query = "MERGE (s:Subreddit {name: $subreddit_name})"
    tx.run(query, subreddit_name=subreddit_name)

# Cypher query to create relationships
def create_posted_by_edge(tx, post_id, user_id):
    query = (
        "MATCH (p:Post {post_id: $post_id}), (u:User {user_id: $user_id}) "
        "MERGE (u)-[:POSTED_BY]->(p)"
    )
    tx.run(query, post_id=post_id, user_id=user_id)

def create_belongs_to_edge(tx, post_id, subreddit_name):
    query = (
        "MATCH (p:Post {post_id: $post_id}), (s:Subreddit {name: $subreddit_name}) "
        "MERGE (p)-[:BELONGS_TO]->(s)"
    )
    tx.run(query, post_id=post_id, subreddit_name=subreddit_name)

# Load and process each chunk file
base_path = r"C:\Users\parth\OneDrive\Desktop\Parth\SJSU\Coursework\297\Dataset\Fakeddit\chunks\merged"
chunk_steps = range(0, 380001, 20000)

for chunk_start in chunk_steps:
    file_name = f"merged_dataset_final_chunk_{chunk_start}.csv"
    file_path = os.path.join(base_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"⚠ Skipping missing file: {file_name}")
        continue

    df = pd.read_csv(file_path)

    print(f"📂 Processing: {file_name} ({len(df)} rows)")
    
    for _, row in df.iterrows():
        try:
            post_data = {
                "post_id": row["post_id"],
                "title": row["cleaned_title"],  # Use cleaned_title instead of title
                #"selftext": row["selftext"],
                #"num_comments": row["num_comments"],
                "upvotes": row["upvotes"],
                "created_utc": row["created_utc"],
                "url": row["url"],
                "label": row["label"],
                "embedding_similarity": row["embedding_similarity"]  # New property
            }

            author_name = str(row["author_id"]) if pd.notna(row["author_id"]) else "deleted_user_" + row["post_id"]
            subreddit_name = str(row["subreddit"]) if pd.notna(row["subreddit"]) else "Unknown"
            user_data = {"user_id": author_name}

            with driver.session() as session:
                session.write_transaction(create_post_node, post_data)
                session.write_transaction(create_user_node, user_data)
                session.write_transaction(create_posted_by_edge, post_data["post_id"], user_data["user_id"])
                session.write_transaction(create_subreddit_node, subreddit_name)
                session.write_transaction(create_belongs_to_edge, post_data["post_id"], subreddit_name)

                print(f"✔ Post {post_data['post_id']} linked to user {user_data['user_id']} and subreddit {subreddit_name}")
        except Exception as e:
            print(f"❌ Error processing post {row['post_id']}: {e}")

print("✅ All chunks processed and loaded to Neo4j.")

# Close the Neo4j driver
driver.close()

'''
// Check Post nodes
MATCH (p:Post) RETURN p LIMIT 10;

// Check User nodes
MATCH (u:User) RETURN u LIMIT 10;

// Check Subreddit nodes
MATCH (s:Subreddit) RETURN s LIMIT 10;

// Check POSTED_BY relationships
MATCH (p:Post)-[]->(u:User) RETURN p, u LIMIT 10;

// Check BELONGS_TO relationships
MATCH (p:Post)-[]->(s:Subreddit) RETURN p, s LIMIT 10;

// Check normal conections
MATCH (n) - [r] -> (m) RETURN n,m,r LIMIT 200

// Delete everything
MATCH (n) DETACH DELETE n
'''






















'''from neo4j import GraphDatabase
import pandas as pd
import os
from config import DATASET_FOLDER, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Establish connection to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Cypher query to create post nodes
def create_post_node(tx, post_data):
    query = (
        "MERGE (p:Post {post_id: $post_id}) "
        "SET p.title = $title, p.selftext = $selftext, p.num_comments = $num_comments, "
        "p.upvotes = $upvotes, p.created_utc = $created_utc, p.url = $url, p.label = $label"
    )
    tx.run(query, **post_data)

# Cypher query to create user nodes
def create_user_node(tx, user_data):
    query = "MERGE (u:User {user_id: $user_id})"
    tx.run(query, **user_data)

# Cypher query to create subreddit nodes
def create_subreddit_node(tx, subreddit_name):
    query = "MERGE (s:Subreddit {name: $subreddit_name})"
    tx.run(query, subreddit_name=subreddit_name)

# Cypher query to create relationships
def create_posted_by_edge(tx, post_id, user_id):
    query = (
        "MATCH (p:Post {post_id: $post_id}), (u:User {user_id: $user_id}) "
        "MERGE (u)-[:POSTED_BY]->(p)"
    )
    tx.run(query, post_id=post_id, user_id=user_id)

def create_belongs_to_edge(tx, post_id, subreddit_name):
    query = (
        "MATCH (p:Post {post_id: $post_id}), (s:Subreddit {name: $subreddit_name}) "
        "MERGE (p)-[:BELONGS_TO]->(s)"
    )
    tx.run(query, post_id=post_id, subreddit_name=subreddit_name)

# Load dataset
data_path = os.path.join(DATASET_FOLDER, "extracted_reddit_data_combined.csv")
df = pd.read_csv(data_path)

# Process each row
for _, row in df.iterrows():
    try:
        post_data = {
            "post_id": row["post_id"],
            "title": row["title"],
            "selftext": row["selftext"],
            "num_comments": row["num_comments"],
            "upvotes": row["upvotes"],
            "created_utc": row["created_utc"],
            "url": row["url"],
            "label": row["label"]
        }
        
        author_name = str(row["author_id"]) if pd.notna(row["author_id"]) else "deleted_user_" + row["post_id"]
        subreddit_name = str(row["subreddit"]) if pd.notna(row["subreddit"]) else "Unknown"
        user_data = {"user_id": author_name}

        with driver.session() as session:
            session.write_transaction(create_post_node, post_data)
            session.write_transaction(create_user_node, user_data)
            session.write_transaction(create_posted_by_edge, post_data["post_id"], user_data["user_id"])
            session.write_transaction(create_subreddit_node, subreddit_name)
            session.write_transaction(create_belongs_to_edge, post_data["post_id"], subreddit_name)

            print(f"✔ Post {post_data['post_id']} linked to user {user_data['user_id']} and subreddit {subreddit_name}")
    except Exception as e:
        print(f"❌ Error processing post {row['post_id']}: {e}")

print("✅ Data loading to Neo4j complete.")

# Close the Neo4j driver
driver.close()



// Check Post nodes
MATCH (p:Post) RETURN p LIMIT 10;

// Check User nodes
MATCH (u:User) RETURN u LIMIT 10;

// Check Subreddit nodes
MATCH (s:Subreddit) RETURN s LIMIT 10;

// Check POSTED_BY relationships
MATCH (p:Post)-[]->(u:User) RETURN p, u LIMIT 10;

// Check BELONGS_TO relationships
MATCH (p:Post)-[]->(s:Subreddit) RETURN p, s LIMIT 10;

// Delete everything
MATCH (n) DETACH DELETE n
'''
