import streamlit as st
from  util.utility import check_password

with st.expander("Important Notice"):
    st.markdown(
        "**IMPORTANT NOTICE:** This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters. Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output. Always consult with qualified professionals for accurate and personalized advice."
    )

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
    
st.title("Methodology")

st.subheader("Use Case 1: Use of RAG")
# display the image
image_path = "app_resources/method1.png"  
st.image(image_path, caption="Methodology Diagram for RAG", use_column_width=True)


st.subheader("Use Case 2: Use of Agent")
# display the image
image_path = "app_resources/method2.png"  
st.image(image_path, caption="Methodology Diagram for Agent", use_column_width=True)

# Value Proposition
st.header("Learnings")
st.write("""
* **Control of Agent:** Compared to RAG, the internal working of Agent seems to result in variance in result on some runs. Observed from verbose, Agent may run multiple or just 1 run even if inputs are the same. This occurs even if the temperture had been set to 0.
* **Crafting of Prompt:** In an attempt to gather comprehensive retrieval, more detailed prompts are added. It could back fire as over-detailed prompt could constraint the result. Encountered that when prompt includes requirement for A and B and C and D, result could be empty.
""")