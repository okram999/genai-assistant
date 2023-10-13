# this is to embed data to Picone.

# use the amazon bedrock embedding to Pinecone
# dimension - 1536 cosine


import os
import boto3

from langchain.document_loaders import S3DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "blogs-embedding-vectors"
S3_BUCKET_NAME = "genai-src-data"
S3_PREFIX = "data"

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)

# i had to locally modify the S3DirectoryLoader class to make it work with the AWS S3 bucket
# https://github.com/langchain-ai/langchain/issues/6535

def load_data():
    client = boto3.client('sts')
    response = client.get_caller_identity()
    print(response)
    loader = S3DirectoryLoader(S3_BUCKET_NAME, prefix=S3_PREFIX)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )
    text = text_splitter.split_documents(document)
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        model_id = "amazon.titan-embed-text-v1"
    )
    docsearch = Pinecone.from_documents(text, embeddings, index_name=INDEX_NAME)
    print("Data loaded to Pinecone")


if __name__ == "__main__":
    load_data()
