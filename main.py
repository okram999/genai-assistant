import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import VectorDBQA, OpenAI


from dotenv import load_dotenv

load_dotenv()


pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


if __name__ == "__main__":

    loader = TextLoader("/Users/niris/Documents/mini-project/blogs/blog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    text = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    docsearch = Pinecone.from_documents(
        text, embeddings, index_name="blogs-embedding-vectors"
    )

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), 
        vectorstore=docsearch, 
        chain_type="stuff"
        # return_source_documents=True
    )

    query = "what is life?"

    result = qa({"query": query})
    print(result)
