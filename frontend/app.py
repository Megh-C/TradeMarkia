import streamlit as st
import requests

API_URL = "http://backend:8000"

st.title("Semantic Search System")

st.write("Search documents using semantic search with caching.")

query = st.text_input("Enter your query:")

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a query.")
    else:

        response = requests.post(
            f"{API_URL}/query",
            json={"query": query}
        )

        data = response.json()

        st.subheader("Result")

        st.write(data["result"])

        st.subheader("Metadata")

        st.write(f"Cluster: {data['cluster']}")

        st.write(f"Cache Hit: {data['cache_hit']}")

        if data["cache_hit"]:
            st.success("Result returned from semantic cache!")