import os

def load_newsgroups(data_dir):

    documents = []
    labels = []
    ids = []

    doc_id = 0

    for category in os.listdir(data_dir):

        category_path = os.path.join(data_dir, category)

        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):

            file_path = os.path.join(category_path, filename)

            with open(file_path, "r", encoding="latin1") as f:
                text = f.read()

            documents.append(text)
            labels.append(category)
            ids.append(str(doc_id))

            doc_id += 1

    return documents, labels, ids