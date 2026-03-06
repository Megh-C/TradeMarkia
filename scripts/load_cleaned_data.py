import json


# load the cleaned dataset that was produced in the preprocessing stage.
#
# The JSON file contains a list of dictionaries with:
#   id      → unique document id
#   label   → original newsgroup category
#   text    → cleaned document text
#
# This function loads the dataset and separates the fields into lists so they can be used for embedding generation and vector db insertion.

def load_cleaned_documents(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    labels = []
    ids = []

    for item in data:

        documents.append(item["text"])
        labels.append(item["label"])
        ids.append(item["id"])

    return documents, labels, ids