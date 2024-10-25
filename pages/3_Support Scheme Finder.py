__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import chromadb.config
import os
import streamlit as st  
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import tool
import crewai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json 
from util.utility import check_password, URLS
from langchain.chains import RetrievalQA

with st.expander("Important Notice"):
    st.markdown(
        "**IMPORTANT NOTICE:** This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters. Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output. Always consult with qualified professionals for accurate and personalized advice."
    )

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
    
# URLs for SkillsFuture support schemes
URLS_SKILLSFUTURE = URLS
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
vector_db_path = 'app_resources/vector_db_2a'

def createRetrival():
    print("Created R1")
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
    
    print("Created R2")
    if not os.path.exists(vector_db_path):
        loader = UnstructuredURLLoader(URLS_SKILLSFUTURE)
        data = loader.load()
        print(data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splitted_documents = text_splitter.split_documents(data)
        vectordb = Chroma.from_documents(splitted_documents, embeddings_model, collection_name='embedding_semantic', persist_directory=vector_db_path)
        print("Created vector db")
    else:
        vectordb = Chroma(collection_name='embedding_semantic', persist_directory=vector_db_path, embedding_function=embeddings_model)
        print("Loading existing vector db")

    return vectordb.as_retriever()

def get_user_input():
    user_data = {}

    # Age selection
    age_options = ["Above 25 and below 40", "Above 40"]
    user_data['age'] = st.radio("Select your age category:", age_options)
    
    # Employment selection
    status_options = ["Employed", "Unemployed"]
    user_data['e_status'] = st.radio("Select your employment status:", status_options)
    
    return user_data

retriever = createRetrival()
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, seed=42)

def main():

    # Define the search tool
    @tool
    def searchtool(query):
        """
        Searches and returns documents regarding the SkillsFuture schemes, initiatives and credits
        Accepts a string as input.
        """

        # Define keywords to include in the search
        keywords = "Find all skillsFuture support, initiatives and credits"

        # Combine the user's query with the predefined keywords
        enhanced_query = f"{keywords} + {query}"
        #query = "Skillfuture Credit, support schemes, initiatives"
        print(f"Query received: {query} (type: {type(query)})")
        docs = retriever.get_relevant_documents(enhanced_query)
        return docs
    

    @tool
    def rag_searchtool(query):
        """
        Searches and returns documents regarding the SkillsFuture schemes, initiatives, and credits.
        Accepts a string as input and generates a response with relevant document sources.
        """

        # Define keywords to include in the search
        keywords = (
            "Identify all Skillfuture scheme and support initiatives or credit"
        )

        # Combine the user's query with the predefined keywords
        enhanced_query = f"{keywords} + {query}"
        print(f"Query received: {enhanced_query} (type: {type(enhanced_query)})")

        # Retrieve relevant documents based on the enhanced query
        docs = retriever.get_relevant_documents(enhanced_query)

        # Set up RetrievalQA to generate an answer based on the retrieved documents
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        response = qa_chain.invoke({"query": enhanced_query})

        # Return both response and relevant documents
        return response

    tools = [searchtool,rag_searchtool]

    agent_SchemeFinder = crewai.Agent(
        role="Scheme Finder",
        goal="You are a know it all on Skillfuture initiatives. Identify all SkillsFuture support schemes, credits, and initiatives.",
        backstory=( 
            "As a Scheme Finder, you are an expert in sourcing and consolidating SkillsFuture's support schemes, credits, and initiatives. "
            "Your aim is to present users with a full list of options avalible, make sure that no relevant scheme is missed. "
            "Include schemes open to all with no eligiability criteria, as well as those tailored to specific user segments."
        ),
        tools=tools,
        allow_delegation=False,
        verbose=True,
    )

    # Define a more detailed and thorough task for scraping schemes
    task_ScrapeSchemes = crewai.Task(
        description=(
            "Perform a detailed scrape of the SkillsFuture website. Extract all support schemes, credits, and initiatives comprehensively. "
            "Capture each scheme's description, eligibility criteria, benefits, and links to resources. If any scheme has no eligibility "
            "criteria or is open to all, specify this explicitly with 'Open to all'"
            "Example for open to all scheme 1. Skillfuture Credit , 2.Skills and Training Advisory"
            "You should retrieve at least 8 schemes, initiatives or support. Take you time. Think step by step"
        ),
        expected_output=(
            "A complete list of SkillsFuture support schemes, including key details, eligibility criteria, benefits, and relevant links. "
            "Schemes open to all or with minimal eligibility should be clearly labeled."
        ),
        agent=agent_SchemeFinder,
        async_execution=True,
    )

    # define agent to check returned info 
    agent_FactChecker = crewai.Agent(
        role="Fact Checker",
        goal="Ensure all information is factually correct and aggregated properly and the user qulifies for the scheme based on their",
        instruction_for_output=(
            "Please return the result as a list of dictionaries. Each dictionary should have the following keys: "
            "'Scheme Name', 'Description', and 'Link'. Ensure the format is easily convertible to a table format."
        ),
        backstory=( 
            "With a keen eye for detail, you ensure that all information about "
            "SkillsFuture schemes is accurate, preventing misinformation and helping "
            "users make informed decisions applying from scheme wherre they meet the eligibility criteria"
        ),
        allow_delegation=False,
        verbose=True,
    )

    # define task for searching
    task_FactCheckAndAggregate = crewai.Task(
        description=(
            "Aggregate and fact-check the scraped schemes and user input against the eligibity criteria"
            "where user age is {user_inputs[age]} years old and employment status is{user_inputs[e_status]} "
            "Includes schemes which Eligibility is Open to all"
        ),
        expected_output=( 
            "The result should return a string in json format without delimiters, that i can run with json.loads, containing two keys: "
            "'summary' - a concise overview summarizing the user input and if any 'eligible_schemes' found, Be informal and enthusiastic and encourage the user to ask more questions!"
            "and 'eligible_schemes' - an array of objects where each object represents an eligible scheme "
            "Each object should include the keys: 'scheme_name', 'description', and 'link'."
        ),
        agent=agent_FactChecker,
        backstory=(
            "Ensure all information on SkillsFuture schemes is accurate, consolidating data into a reliable, structured JSON report "
            "for users. Eliminate errors and outdated information for clear, fact-based recommendations."
        ),
        context=[task_ScrapeSchemes],
    )

    # crew creation
    crew = crewai.Crew(
        agents=[agent_SchemeFinder, agent_FactChecker],
        tasks=[task_ScrapeSchemes, task_FactCheckAndAggregate],
        verbose=True,
        llm=llm,
        cache=True,
    )
        
    st.title("Support Scheme Finder")

    with st.form("my_form2"):
        user_inputs = get_user_input()
        
        submitted = st.form_submit_button("Submit Profile")
    
    if submitted:
        with st.spinner('Processing...'):
    
            print(user_inputs)
                # Wrap user inputs for task interpolation
            wrapped_user_inputs = {
                'user_inputs': {
                    'age': user_inputs['age'],  # Ensure correct key usage
                    'e_status': user_inputs['e_status']  # Ensure correct key usage
                }
            }
            # Kickoff the FactCheck task with wrapped user inputs
            result = crew.kickoff(inputs=wrapped_user_inputs)
            
            # Load the raw JSON string directly into a Python dictionary
            data = json.loads(result.raw)
            print(f"Data loaded from raw JSON: {data}")  # For debugging
            
            # Check if the expected key exists in the loaded data
            if not data['eligible_schemes']:
                st.write(data['summary'])
            else:
                st.write(data["summary"])
                # Convert the 'eligible_schemes' into a DataFrame
                df = pd.DataFrame(data["eligible_schemes"])
                
                df.rename(columns={
                    "scheme_name": "Scheme Name",
                    "description": "Description",
                    "link": "More Info"
                }, inplace=True)
                st.header("Overview of Eligible SkillsFuture Schemes")
                
                st.table(df[['Scheme Name', 'Description']])

                # Display DataFrame with multi-line and hyperlinks
                st.markdown("### Schemes Links")
                for index, row in df.iterrows():
                    st.markdown(f"**{row['Scheme Name']}**")
                    st.markdown(f"{row['Description']}")
                    st.markdown(f"[More Info]({row['More Info']})")  # Change to hyperlink
                    #st.markdown("---")  # Separator for each scheme
            

# End of main script
if __name__ == "__main__":
    main()
