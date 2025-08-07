import sys
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
import ollama
from langchain_chroma import Chroma
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from dbwriter import DBWriter
from pdfingestor import PDFIngestor
from flask import Flask, render_template, request

ACTIVE_LLM = "qwen3:14b"
ConversationModel = OllamaLLM(model=ACTIVE_LLM, temperature=0.3)

def ingest_book_pdf(filename: str):
    if DBWriter.db_needs_generated():
        content = PDFIngestor.read_pdf_file(filename)
        content = PDFIngestor.clean_extracted_text(content)
        chunks = PDFIngestor.chunk_text(content)
        embedded_texts = DBWriter.static_ollama_embeddings.embed_documents(chunks)
        ids = [f"{i}" for i in range(len(chunks))]
        DBWriter.get_collection().add(ids, embeddings=embedded_texts, documents=chunks)

prompt = """
You are a helpful AI roleplaying game expert system that answers questions based on the provided context.
Rules:
1. Only use information from the provided context to answer questions except when asked to perform tasks specifically requiring creativity such as generating adventures or character biographies.
2. If the context doesn't contain enough information, say so honestly
3. Be specific and cite relevant parts of the context
4. Keep your answers clear and concise
5. If you're unsure, admit it rather than guessing
Context:
{context}
Question: {input}
Answer based on the context above:
"""
prompt_template = ChatPromptTemplate.from_template(prompt)
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
ingest_book_pdf(sys.argv[1])
rag_chain = ({
        'context': DBWriter.get_retriever(),
        'input': RunnablePassthrough()
    } | prompt_template | ConversationModel | StrOutputParser()
    )

app = Flask(__name__)
@app.get('/')
def show_form():
    return render_template('rag_interface.html', response="")

@app.post('/')
def submit():
    text_input = request.form.get('user_input')
    return run_agent(text_input)

def run_agent(user_input):
        response = rag_chain.invoke(user_input)
        return render_template('rag_interface.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)

