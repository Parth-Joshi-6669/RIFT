# RIFT
RIFT - Reddit Information Falsity Tagger


**Initial setup:**
1) Make sure the related TSV files (fakeddit dataset) are setup on your system.
2) Create a Reddit account and request for API access, you will need ClientID, ClientSecret and UserAgent information.
3) Setup Neo4J on your system. Create a new project and start a database. You will require Neo4J URI, User and Password.
   
**Python Script run sequence:**
Once ready, go to the code folder and run the files in the following order -
1. fakeddit_extract.py (This helps us extract realtime information from Fakeddit dataset and start creating the base for our FakedditEnhanced dataset)
2. mergecsvnstuff.py (For combining the chunks of extracted data)
3. similarity_calc.py (For creating the embedding_similarity information)
4. load_to_neo4j.py (Loading the information into the Neo4J database. Make sure the DB is running when running this script)
5. gnn_train_GCN.py (training the GCN model)
6. gnn_train_graphsage.py (training the GraphSAGE model)

You can edit the code and use the trained model for predicting Fake news on different posts (^_^)
