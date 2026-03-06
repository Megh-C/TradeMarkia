from load_dataset import load_newsgroups
from clean_text import clean_document
from tqdm import tqdm
import json

DATA_DIR = "data/raw/20_newsgroups"


def preprocess():

    documents, labels, ids = load_newsgroups(DATA_DIR)

    cleaned_docs = []

    for doc in tqdm(documents):
        cleaned_docs.append(clean_document(doc))

    return cleaned_docs, labels, ids


if __name__ == "__main__":

    docs, labels, ids = preprocess()

    print("Total documents:", len(docs))

    cleaned_dataset = []

    for doc, label, doc_id in zip(docs, labels, ids):

        cleaned_dataset.append({
            "id": doc_id,
            "label": label,
            "text": doc
        })

    # Save cleaned documents to disk
    with open("data/processed/cleaned_documents.json", "w", encoding="utf-8") as f:

        json.dump(cleaned_dataset, f, indent=2)