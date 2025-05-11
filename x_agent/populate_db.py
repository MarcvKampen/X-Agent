import csv
import logging
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
# Assuming tweets.csv is in the same directory as this script (x_agent/)
CSV_FILE_PATH = "tweets.csv"
DB_DIRECTORY = (
    "chroma_db"  # Will be created in the same directory as this script (x_agent/)
)
COLLECTION_NAME = "tweets"
# Use the same embedding model as in agent_core.py for consistency
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# --- End Configuration ---


def load_tweets_from_csv(file_path):
    """Loads tweets from a CSV file."""
    tweets = []
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
            # Assuming pipe-delimited based on previous discussions for tweets.csv
            # and that tweets are in the first column.
            reader = csv.reader(csvfile, delimiter="|", quoting=csv.QUOTE_NONE)
            for i, row in enumerate(reader):
                if row and row[0].strip():
                    tweets.append(row[0].strip())
                elif row and not row[0].strip():
                    logging.warning(f"Skipping empty tweet in CSV at row {i + 1}")
                elif not row:
                    logging.warning(f"Skipping empty row in CSV at row {i + 1}")

        logging.info(f"Successfully loaded {len(tweets)} tweets from {file_path}")
        return tweets
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading CSV file {file_path}: {e}")
        return []


def initialize_db(db_path, collection_name):
    """Initializes ChromaDB client and gets/creates a collection."""
    try:
        client = chromadb.Client(
            Settings(persist_directory=db_path, is_persistent=True)
        )
        # get_or_create_collection will create if not exists, or get if it does.
        collection = client.get_or_create_collection(
            name=collection_name,
            # Optionally, specify metadata for the embedding function if needed by ChromaDB version
            # For SentenceTransformer, usually not needed here as embeddings are pre-calculated.
            # metadata={"hnsw:space": "cosine"} # Example if you want to specify distance metric
        )
        logging.info(
            f"ChromaDB client initialized. Using/creating collection '{collection_name}' at {db_path}"
        )
        return collection
    except Exception as e:
        logging.error(f"Error initializing ChromaDB: {e}")
        return None


def populate_vector_db(collection, tweets, model):
    """Generates embeddings and populates the ChromaDB collection."""
    if not tweets:  # model being None is handled before this call
        logging.warning("No tweets to process.")
        return
    if collection is None:
        logging.error("DB collection not available. Cannot populate.")
        return

    logging.info(
        f"Generating embeddings for {len(tweets)} tweets using '{EMBEDDING_MODEL_NAME}'..."
    )
    try:
        # It's good practice to ensure the collection is empty before repopulating if this script is run multiple times
        # to avoid duplicate IDs, or use add with specific IDs that are idempotent.
        # For simplicity here, we assume either a fresh DB or that IDs are unique if re-running.
        # A more robust approach might be to use `collection.upsert`.

        # Check if collection has items and warn or clear if necessary
        if collection.count() > 0:
            logging.warning(
                f"Collection '{collection.name}' already contains {collection.count()} items. "
                "If re-populating, consider clearing it first or using upsert to avoid duplicates/errors."
            )
            # Example: For a full refresh, you might delete and recreate the collection or delete items.
            # client.delete_collection(name=COLLECTION_NAME)
            # collection = client.create_collection(name=COLLECTION_NAME)
            # This script will proceed with adding, which might lead to duplicate ID errors if IDs are not unique
            # or just add more items if IDs are unique (e.g. based on new CSV content).

        embeddings = model.encode(tweets, show_progress_bar=True)
        logging.info("Embeddings generated successfully.")

        ids = [f"tweet_{i}" for i in range(len(tweets))]  # Simple unique IDs
        metadatas = [{"text": tweet} for tweet in tweets]

        logging.info(
            f"Adding {len(ids)} items to ChromaDB collection '{collection.name}'..."
        )
        collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
        logging.info("Successfully added tweets and embeddings to the database.")
        logging.info(
            f"Database collection '{collection.name}' now contains {collection.count()} items."
        )

    except Exception as e:
        logging.error(f"Error during embedding generation or DB population: {e}")


if __name__ == "__main__":
    logging.info("--- Starting Vector DB Population Script ---")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_full_path = os.path.join(script_dir, CSV_FILE_PATH)
    db_full_path = os.path.join(
        script_dir, DB_DIRECTORY
    )  # DB will be in x_agent/chroma_db

    logging.info(
        f"Attempting to load sentence transformer model: {EMBEDDING_MODEL_NAME}..."
    )
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Sentence transformer model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load sentence transformer model: {e}")
        embedding_model = None

    if embedding_model:
        tweets_list = load_tweets_from_csv(csv_full_path)

        if tweets_list:
            tweet_collection = initialize_db(db_full_path, COLLECTION_NAME)

            if tweet_collection:
                populate_vector_db(tweet_collection, tweets_list, embedding_model)
        else:
            logging.warning("No tweets loaded from CSV. DB will not be populated.")
    else:
        logging.error("Embedding model could not be loaded. DB population aborted.")

    logging.info("--- Vector DB Population Script Finished ---")
