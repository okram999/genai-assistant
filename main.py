import os
from typing import Set
from lib.backend import ask_question
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string



st.header("My Research - Generative AI Assistant")
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
        generated_response = ask_question(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        # hard code metadata prefix to trim the source path.
        # This needs to be fixed. use it as a variable!
          
        sources = set(
            [doc.metadata["source"].lstrip('/var/folders/2c/zg0dqdp129v0bz0s7l00k3vm0000gr/T/tmp3f34f9og/') for doc in generated_response["source_documents"]]
        )
        
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )


        st.session_state.chat_history.append((prompt, generated_response["answer"]))
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(formatted_response)

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
