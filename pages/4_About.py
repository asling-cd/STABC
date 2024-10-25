import streamlit as st

from util.utility import check_password

with st.expander("Important Notice"):
    st.markdown(
        "**IMPORTANT NOTICE:** This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters. Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output. Always consult with qualified professionals for accurate and personalized advice."
    )

#Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
    
st.title("About")

# Project Overview
st.header("Project Overview")
st.write("""
Explore how different LLM techniques can help users make informed decisions about future skills and available support schemes.
""")

# Value Proposition
st.header("Objectives")
st.write("""
* **Understand Future Skills Demand:** Access summaries and answers to specific questions about the Skills Demand for the Future Economy 2023/24 report. This report is a valuable resource but can be difficult to read due to its complex data, methodology, and calculations. We use AI to:
    * Condense complex information into easy-to-understand insights.
    * Provide personalized responses.

* **Navigate Support Schemes:** Easily identify and apply for relevant support schemes based on user profile.
""")

st.header("Features")
st.write("""
* **Understand Future Skills Demand:**
    * Uses Retrieval-Augmented Generation (RAG) to understand if users are asking general or specific questions.
    * Based on the question type, tailored prompts are crafted to answer users queries effectively.
    * Implemented methodologies including multiple queries and ranking to improve the response quality.

* **Navigate Support Schemes:**
    * Use an AI Agent approach:
        * One agent analyzes SkillFuture initiatives scraped from relevant websites.
        * Another agent fact-checks the information and identifies relevant schemes based on users input.

**Note:** Two different techniques are deliberately planned to explore the pros and cons through practical implementation.
""")

st.header("Security")
st.write("""
* **Prompt Injection Prevention Using {{}} :**
    * Placeholders are used to mitigate the risk of prompt injection.
    * Prompts are crafted with sufficient background and context to lower risk of responding to malicous input.
""")

st.header("Data Sources")
st.write("""
* Report on Skills Demand for the Future Economy 2023/24
* 20 different URLs from "https://www.skillsfuture.gov.sg" and "https://www.myskillsfuture.gov.sg"
""")

st.header("TODO")
st.write("""

* **Change of Approach for Skill Demand Understanding:**
  - Consider using an agent-based approach instead of pure RAG. 
  - A pure RAG approach requires careful crafting of multiple prompts to emulate a thought process, which can be less efficient than an agent-based implementation.
  - An agent can identify the query type (general/job search) and route it to the appropriate agent.
  - Additional agents can also be added for course and job searches.

* **Integrate Real-time Job Market Data:**
  - Provide up-to-date insights into industry trends.

* **Implement Course Recommendation:**
  - Recommend relevant skills and courses based on user interests.

* **Expand Language Support:**
  - Leverage LLM to perform translations.
""")