# this is to embed data to Picone.

import os

from langchain.document_loaders import S3FileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "blogs-embedding-vectors"

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def load_data():
    loader = S3FileLoader("genaiassistant", "diabetes.txt")
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(text, embeddings, index_name=INDEX_NAME)
    print("Data loaded to Pinecone")


if __name__ == "__main__":
    load_data()
