import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain import VectorDBQA, OpenAI
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()


pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)


# UI code below

st.header("XYZ Research - Generative AI Assistant")


# backend code below

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

    # query = "what is life?"

    # result = qa({"query": query})
    # print(result)

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


prompt = st.text_input(
    "Prompt", placeholder="Ask me about your test results..."
) or st.button("Submit")

if prompt:
    with st.spinner("Asking AI..."):
        generated_response = qa({"query": prompt})
        resp = generated_response["result"]
        # st.session_state["chat_answers_history"].append(resp)

        st.session_state.chat_history.append((prompt, generated_response["result"]))
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(resp)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(
            user_query,
            is_user=True,
        )
        message(generated_response)
