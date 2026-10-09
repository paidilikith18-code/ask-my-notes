from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load document
loader = TextLoader("data/notes/ai.txt")
documents = loader.load()

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

# Split document
chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}")
    print("Length:", len(chunk.page_content))
    print("Content:", chunk.page_content)