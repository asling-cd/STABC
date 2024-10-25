import streamlit as st  
import random  
import hmac  
from  util.utility import check_password

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

# Main content
st.header('Home')

# Description of the application
st.write("""
This web application hosts 2 functionalities designed to help you navigate the future skills demand and available support schemes for skilling.
""")

# Application sections
st.header("Applications")

# Section for Future Skills Demand
st.subheader("1. Understand Future Skills Demand")
st.write("""
* **Understand Future Skills Demand:** Access summaries and answers to specific questions about the Skills Demand for the Future Economy 2023/24 report. This report is a valuable resource but can be difficult to read due to its complex data, methodology, and calculations. We use AI to:
    * Condense complex information into easy-to-understand insights.
    * Provide personalized responses.
""")

# Section for Support Schemes
st.subheader("2. Navigate Support Schemes")
st.write("""
* **Navigate Support Schemes:** Easily identify and apply for relevant support schemes based on your profile.
""")

st.write("""
### Get Started
Select the application you wish to explore from the sidebar to begin your journey of future proofing your skills!
""")