__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from langchain.vectorstores import Chroma
import requests
import PyPDF2
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.schema import Document
import warnings
from langchain.globals import set_verbose, set_debug
import logging
from util.utility import check_password


with st.expander("Important Notice"):
    st.markdown(
        "**IMPORTANT NOTICE:** This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters. Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output. Always consult with qualified professionals for accurate and personalized advice."
    )

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

#Dev config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)  # Enable all logging levels
#logging.getLogger().setLevel(logging.CRITICAL) # Disable logging
#set_verbose(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#App variable Init

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]


# Embeddings and LLM initialization
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, seed=42)

# PDF URL and file path
pdf_url = "https://www.skillsfuture.gov.sg/docs/default-source/skills-report-2023/sdfe-2023.pdf"
pdf_file_path = "app_resources/sdfe-2023.pdf"
vector_db_path = 'app_resources/vector_db_1'

def download_pdf():
    response = requests.get(pdf_url)
    with open(pdf_file_path, "wb") as pdf_file:
        pdf_file.write(response.content)
    logging.info("New File Downloaded")
def extract_documents():
    logging.info("Extracting documents")
    documents = []
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                document = Document(page_content=page_text, metadata={"page_number": page_number})
                documents.append(document)
    return documents

def create_vector_db():
    if not os.path.exists(vector_db_path):
        # Download PDF and extract documents
        download_pdf()
        
        documents = extract_documents()
        # Split documents and create vector database
        text_splitter = SemanticChunker(embeddings_model)
        splitted_documents = text_splitter.split_documents(documents)
        logging.info("Initialising vector db")
        vectordb = Chroma.from_documents(splitted_documents, embeddings_model, collection_name='embedding_semantic', persist_directory=vector_db_path)
        logging.info("Created vector db")
        vectordb.persist()
    else:
        # Load existing vector database
        vectordb = Chroma(collection_name='embedding_semantic', persist_directory=vector_db_path, embedding_function=embeddings_model)
        logging.info("Loading existing vector db")
    return vectordb

def setup_retrievers(vectordb):
    logging.info("Running Retriever")
    template = """
        Your task is to generate 3 distinct search queries based on the user-provided keywords.
        The keywords relate to job searches within the user’s specified interests, skills, and career goals.
        Make sure to include synonyms and related terms to broaden the search scope.
        If the provided keywords don’t align with current trends or are unclear, suggest related fields or popular job roles that are currently in demand based on recent labor market data.
        Each query MUST approach the keywords from a different angle to maximize diversity and relevance in the results.
        Present these alternative queries clearly, each on a new line for easy readability.

        **Example Keywords**: "data science, machine learning, AI"

        **Output Format**:
        1. 
        2. 
        3. 

        Original question: {question}
        """

    query_prompt_template = PromptTemplate(input_variables=["user_prompt"], template=template)
    retriever_multiquery = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm, prompt=query_prompt_template)
    
    #compressor = CohereRerank(top_n=3, model='rerank-english-v3.0')
    #compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_multiquery)
    
    #return compression_retriever
    return retriever_multiquery

def setup_retrievers_general(vectordb):
    logging.info("Running Retriever General")
    template = """
        Your task is to answer the user question with regards to the future job market.
       
        Original question: {question}
        """

    query_prompt_template = PromptTemplate(input_variables=["user_prompt"], template=template)
    retriever_multiquery = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm, prompt=query_prompt_template)
    
    compressor = CohereRerank(top_n=3, model='rerank-english-v3.0')
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_multiquery)
    
    return compression_retriever
    #return retriever_multiquery

def query_job_market(user_prompt, retriever):
    logging.info("Query Job Market")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    response = qa_chain.invoke({"query": user_prompt})
    return response

def query_general_market(user_prompt, retriever):
    logging.info("Query General Market")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    response = qa_chain.invoke({"query": user_prompt})
    return response

def extract_keywords_with_llm(user_query):
   
    prompt = f"""
    Determine if the following user query is a general question or focused on skills, interests, or job inquiries.
    If it is focused, extract the relevant keywords and return them as a comma-separated string.
    If it is a general question, return an empty string. 

    Please ensure that the output is exactly as follows:
    - If the query is general, return an empty string: ""
    - If the query is focused, return the keywords as a comma-separated string.

    User query: "{user_query}"
    """

    # simple zero shot prompt
    response = llm(prompt)  

    content = response.content  

    # remove spaces
    return content.strip()


def main():

    vectordb = create_vector_db()

    retriever = setup_retrievers(vectordb)
    retriever_general = setup_retrievers_general(vectordb)

    #result = extract_keywords_with_llm("What are the in demand economy ?")
    #print(f"Extracted keywords: '{result}'\n")

    st.title("Explore Your Future Career!")
    st.sidebar.title("Input Ideas")
    st.sidebar.markdown("""
            What is green economy?  
            Cybersecurity job  
            I am exploring nursing
            """)
    with st.form("my_form"):
        user_input = st.text_input(
            f"Share your interests, skills, or goals, and I’ll help you discover in-demand job opportunities that suit you."
            f"You can also ask me anything about the future job market!"
            )
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        with st.spinner('Processing...'):
            # Execute logic only when button is pressed

            if user_input:
                result = extract_keywords_with_llm(user_input)
                print(f"User query: '{user_input}'\nExtracted keywords: '{result}'\n")
               
                if result == '""':
                    print("The string is empty.")

                    user_prompt = (
                        f"You are job market expert with deep expertise on the skills demand of future economy."
                        f"Be informal and enthusiastic and encourage the user to ask more questions!"
                        f"Answer the user queries, provided in {user_input}." 
                        f"Think step by step."
                    )
                    response = query_general_market(user_prompt, retriever_general)
                else:
                    print("The string is not empty.")
                    user_prompt = (
                        f"You are job market expert with deep expertise on the skills demand of future economy."
                        f"Be informal and enthusiastic and encourage the user to ask more questions!"
                        f"Find an in-demand job that aligns with the user's interests or skills or goals, provided in {user_input}." 
                        f"Provide the relevant job recommendations, in demand economy sector "
                        f", the skills needed to pursue them and other info that are avalible such as salary, training etc."
                        f"If you dont know or the user's input does not align with current demands, highlight the limitation such as" 
                        f"while it may not be a high-demand role,it doesn't mean the job is disappearing. "
                        f"Suggest in-demand industries for the user to explore. "
                    )
    
                    response = query_job_market(user_prompt, retriever)

                st.write("Answer:", response['result'])
                #st.write("Source Documents:", response['source_documents'])

if __name__ == "__main__":
    main()
