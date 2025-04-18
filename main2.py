import os
import streamlit as st
from services.mainServices import MainServices


main_serv = MainServices()

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    accept_multiple_files=False,
    label_visibility="hidden"
)

# st.button("Process PDF", on_click=load_pdf, args=(pdf_directory + pdf_name,))

if uploaded_file:
    main_serv.upload_pdf(uploaded_file)
    docs = main_serv.load_pdf(uploaded_file.name)
    chunked_docs = main_serv.split_text(docs)
    main_serv.index_documents(chunked_docs)

    question = st.chat_input("Ask a question about the PDF:")

    if question:
        st.chat_message("user").write(question)
        retrieved_docs = main_serv.retrieve_documents(question)
        answer = main_serv.answer_question(question, retrieved_docs)

        st.chat_message("assistant").write(answer)