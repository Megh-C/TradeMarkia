import chromadb


from load_cleaned_data import load_cleaned_documents
from embed_documents import load_model, embed_documents


DATA_PATH = "data/processed/cleaned_documents.json"


def build_database():

    documents, labels, ids = load_cleaned_documents(DATA_PATH)

    model = load_model()

    embeddings = embed_documents(model, documents)

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.create_collection(
        name="newsgroups"
    )

    # ---------------------------------------------------------
    # Insert documents in batches to avoid exceeding ChromaDB
    # internal batch size limits.
    # ---------------------------------------------------------

    batch_size = 500

    for i in range(0, len(documents), batch_size):

        batch_docs = documents[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings.tolist(),
            metadatas=[{"label": l} for l in batch_labels],
            ids=batch_ids
        )

        print(f"Inserted batch {i} to {i + len(batch_docs)}")


    print("Vector database created successfully.")


if __name__ == "__main__":
    build_database()