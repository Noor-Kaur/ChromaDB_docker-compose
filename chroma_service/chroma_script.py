import chromadb
from chromadb.config import Settings
import requests

def health_check():
    url = 'http://localhost:5000/health'
    resp = requests.get(url)
    return resp.content

def embed_func(texts: list):
    url = 'http://localhost:5000/v1/embed'
    resp = requests.post(url, json=texts)
    print(resp.status_code)
    return resp.json()

# We might then want to run it in client-server mode, where we have a remote server running and then connect to it using an HTTP request.
client = chromadb.HttpClient(host='localhost', port=8000)

collection = client.create_collection(name="store", embedding_function=embed_func, get_or_create=True)

collection.add(
    documents=['dog', 'cat', 'planet', 'moon'],
    ids=[f"id{i}" for i in range(4)]
)

print(collection.get())
print(embed_func(['dog', 'cat', 'planet', 'moon']))
results = collection.query(
    query_texts=["What is the animal name?"],
    n_results= 2
)
print(results)

