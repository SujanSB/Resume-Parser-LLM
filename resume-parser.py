import re
import os
import tempfile
import pinecone
from langchain.vectorstores import Pinecone

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
# from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.prompts.prompt import PromptTemplate
# from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

huggingface_api_token = os.getenv("HUGGINGGACE_API_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def initialize_session_state():
    """
    initialization of session state
    """
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

def parsing_conv_chat(query, chain, history):
    """
    Funciton to generated answer passing through chain 
    and perform some processing before returning results.
    """
    result = chain.run(query)
    lines = result.strip().split('\n')
    match = re.search(r'Answer:(.*?)Question:', result, re.DOTALL)

    first_answer = ""
    rest_of_questions = []

    i = 0
    while i < len(lines):
        if lines[i].startswith('Answer'):
            if match:
                first_answer = match.group(1).strip()
            else:
                first_answer = lines[i][8:]
            i += 1
            while i < len(lines) and not lines[i].startswith('Question'):
                i += 1
        elif lines[i].startswith('Question'):
            rest_of_questions.append(lines[i][10:])
            i += 1
        else:
            i += 1

    history.append((query, first_answer))

    return first_answer


user_input = st.chat_input("Ask anything...")

def display_parsing_history(chain):
    """
    Funciton to display old questions and generated answer.
    """
    reply_container = st.container()
    container = st.container()
    with container:
        # user_input = st.chat_input("Ask anything...")
        if user_input:
            with st.spinner('Generating response...'):
                output = parsing_conv_chat(
                    user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                # Display user's input
                with st.chat_message(name='User'):
                    st.write(st.session_state["past"][i])

                # Display generated response
                with st.chat_message(name="RP"):
                    st.write(st.session_state["generated"][i])

def create_retrieval_chain(vector_store):
    '''
    With some standard prompting this funciton returns
    the chain which made with mistrailai model.
    '''
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"

    llm = HuggingFaceHub(huggingfacehub_api_token=huggingface_api_token,
                         repo_id=model_id,

                         model_kwargs={"temperature": 0.1,
                                       "max_new_tokens": 200})
    template = """
    I want you to act as a Hiring Manager for a Tech Company.
    Your job is to extract the relevant information from the  and provide answer to the user about the candidate.
    The name of the candidate is given at top part of the document, so find the name and if the name is asked then return the name.
    For Education related question, get answer from the Education section,
    For projects related questions, get answer from the projects section,
    For certificates related questions, get answer from certification or certificates section,
    For work experience related questions, get answer from work experience or experience section,
    For technical Skills related question get answer from technical Skills or skills section,
    There may not exist any of above section then return "Relevent information is not provided".
    
    Context: {context}

    Question: {question}

    """

    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )
    return qa

def resume_parser():
    '''
    Function that contain mainly ui part and function call.
    '''
    # Initialize session state

    st.markdown(
        """
    <style>
    .title {
        text-align: center;
        margin-bottom: 20px;
        font-size:60px;
    }
    </style>
    """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='title'>Resume Parser</h1>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload the Resumes here", accept_multiple_files=True)
    # count = 0

    if uploaded_files:
        initialize_session_state()

        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        separator = " ",
        length_function=len
        )

        text_chunks = text_splitter.split_documents(text)
        # num_pages = len(text_chunks)
        # print(num_pages)
        # print(text[1].page_content)
        # breakpoint()

        # Create embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        # embeddings = OpenAIEmbeddings(api_key=openai_api_key)

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Create vector store
        pinecone.init(
            api_key=pinecone_api_key,
            environment="gcp-starter"
        )
        index_name = "resume-parser"

        vector_store = Pinecone.from_documents(text_chunks, embeddings, index_name=index_name)
        chain = create_retrieval_chain(vector_store)

        st.markdown(
            """
                <div style='border-top: 2px solid #f2f2f2;
                margin:20px; border-radius: 1px;'>
                """,
            unsafe_allow_html=True
        )

        with st.chat_message(name="RP"):
            st.write("Start Parsing The Uploaded Resume.")
        display_parsing_history(chain)

        st.markdown(
            """
                </div>
                """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    resume_parser()
