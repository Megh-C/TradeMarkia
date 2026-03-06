import re



# Remove email-style headers from the beginning of posts
# example:
# From:
# Subject:
# Organization:
# Date:
# Message-ID:
#
# the header section ends at the first blank line.
# as this is just metadata we remove all this and the logic is basically removing everything before the first blank line 
# as all that is header which is not relevant to us 
# ---------------------------------------------------------
def remove_headers(text):

    parts = text.split("\n\n", 1)

    if len(parts) > 1:
        return parts[1]

    return text


# Remove quoted reply lines
#
# users quote previous messages
# using the ">" prefix. Example:
#
# > Ex: someone wrote:
# > guns should be banned
#
# Keeping these would duplicate older messages across many documents and increase similarity between unrelated responses.
# so we remove all lines beginning with ">".
# ---------------------------------------------------------
def remove_quotes(text):

    lines = text.split("\n")

    cleaned = []

    for line in lines:
        if not line.strip().startswith(">"):
            cleaned.append(line)

    return "\n".join(cleaned)


# Remove thread reference lines
#
# In article <123@server> someone writes:
# they do not contribute to semantic topic understanding and often include email addresses and message IDs.
#
# the logic is removing lines starting with "In article".
# ---------------------------------------------------------
def remove_article_references(text):

    lines = text.split("\n")

    cleaned = []

    for line in lines:

        if not line.strip().lower().startswith("in article"):
            cleaned.append(line)

    return "\n".join(cleaned)


# Remove email addresses
#
# email addresses frequently appear in headers,references, and signatures.
# remove them using a regex pattern.
# -------------------------------------
def remove_emails(text):

    email_pattern = r"\S+@\S+"

    return re.sub(email_pattern, "", text)



# Remove signature blocks
# Example:
# John Smith
# MIT AI Lab
# ---------
# we remove everything after the signature line as that is basically noise and we dont need to know who wrote that 
# assuming the authors are needed we can omit this part but right now i am making the assumption that the body 
# of the posts is the only thing which is relevant 

def remove_signatures(text):

    signature_pattern = r"\n--\s*\n.*"

    return re.sub(signature_pattern, "", text, flags=re.DOTALL)


def remove_metadata_lines(text):

    lines = text.split("\n")

    cleaned = []

    for line in lines:

        lower = line.lower().strip()

        # remove dataset metadata lines
        if lower.startswith("archive-name"):
            continue

        if lower.startswith("alt-atheism-archive-name"):
            continue

        if lower.startswith("last-modified"):
            continue

        if lower.startswith("version"):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# main cleaning pipeline
# applies all preprocessing steps sequentially 

def clean_document(text):

    text = remove_headers(text)

    text = remove_quotes(text)

    text = remove_article_references(text)

    text = remove_emails(text)

    text = remove_signatures(text)

    text = remove_metadata_lines(text)

    text = text.strip()

    return text