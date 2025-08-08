import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM
import ollama
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from dbwriter import DBWriter
from pdfingestor import PDFIngestor
from flask import Flask, flash, render_template, request, redirect, session
from werkzeug.utils import secure_filename
import secrets

ACTIVE_LLM = "qwen3:14b"
ConversationModel = OllamaLLM(model=ACTIVE_LLM, temperature=0.3)

def ingest_book_pdf(filename: str):
    if not DBWriter.does_collection_exist_for_pdf(filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        content = PDFIngestor.read_pdf_file(filepath)
        content = PDFIngestor.clean_extracted_text(content)
        chunks = PDFIngestor.chunk_text(content)
        embedded_texts = DBWriter.static_ollama_embeddings.embed_documents(chunks)
        ids = [f"{i}" for i in range(len(chunks))]
        DBWriter.get_collection_for_pdf(filename).add(ids, embeddings=embedded_texts, documents=chunks)

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"pdf"}

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
secret_key = secrets.token_hex(16)
app.config['SECRET_KEY'] = secret_key
app.config.from_object(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.get('/')
def show_base_form():
    if 'pdf_filename' in session and session['pdf_filename'] is not None:
        return render_template('rag_interface.html', response="", conversation_visibility=True, upload_visiblity=False)
    return render_template('rag_interface.html', response="", conversation_visibility=False, upload_visiblity=True)

@app.post('/')
def submit():
    if session['pdf_filename'] is None:
        return show_base_form()
    pdf_filename = session['pdf_filename']
    text_input = request.form.get('user_input')
    return run_agent(text_input, pdf_filename)

def run_agent(user_input, pdf_filename: str):
    rag_chain = ({
                     'context': DBWriter.get_retriever_for_pdf(pdf_filename),
                     'input': RunnablePassthrough()
                 } | prompt_template | ConversationModel | StrOutputParser()
                 )
    response = rag_chain.invoke(user_input)
    return render_template('rag_interface.html', response=response)

@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['pdf_filename'] = filename
            ingest_book_pdf(file.filename)
            return redirect('/')
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
    <input type=file name=file>
    <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)

