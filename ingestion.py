import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core import embeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()
MODEL="qwen3-embedding:0.6b"

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("/Users/gkg/Documents/GitHub/personal/Langchain/langchain-course/mediumblog1.txt")
    document= loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings= OllamaEmbeddings(model=MODEL)

    print("ingesting to pinecone...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")