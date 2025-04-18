import os, sys
import streamlit as st
from services.mainServices import MainServices

main_serv = MainServices()

# input from streamlit for user pick what kind of model use
model_choice = st.selectbox(
    "Select the model to use:",
    ("Online", "Local")
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    accept_multiple_files=False,
    label_visibility="hidden"
)

# question = st.text_input("Ask a question about the PDF:")
# st.button("Process PDF", on_click=load_pdf, args=(pdf_directory + pdf_name,))

if uploaded_file:
    main_serv.upload_pdf(uploaded_file)
    docs = main_serv.load_pdf(uploaded_file.name) # pdf_directory (internal) + uploaded_file.name
    chunked_docs = main_serv.split_text(docs)
    main_serv.index_documents(chunked_docs, model_choice)

    question = st.chat_input("Ask a question about the PDF:")

    if question:
        st.chat_message("user").write(question)
        retrieved_docs = main_serv.retrieve_documents(question, model_choice)
        answer = main_serv.answer_question(question, retrieved_docs, model_choice)

        st.chat_message("assistant").write(answer)