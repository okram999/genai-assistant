import os
from lib.backend import ask_question
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

st.header("XYZ Research - Generative AI Assistant")

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
        resp = generated_response["answer"]

        st.session_state.chat_history.append((prompt, generated_response["answer"]))
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
