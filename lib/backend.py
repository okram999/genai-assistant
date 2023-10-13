import os
from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

from dotenv import load_dotenv


load_dotenv()

INDEX_NAME = "blogs-embedding-vectors"

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


def get_llm():
        
    model_kwargs =  { #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": ["Human:"], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        model_id="ai21.j2-ultra-v1", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm



def ask_question(query: str, chat_history: List[Dict[str, Any]] = []):
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        model_id = "amazon.titan-embed-text-v1"
    )
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    chat = get_llm()


    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})
