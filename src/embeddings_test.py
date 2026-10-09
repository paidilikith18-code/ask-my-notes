from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

text = "Python is a programming language."

vector = embeddings.embed_query(text)

print("Vector length:", len(vector))
print("First 10 values:", vector[:10])