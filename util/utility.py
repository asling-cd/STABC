# filename: utility.py
import streamlit as st  
import random  
import hmac  
  
# """  
# This file contains the common components used in the Streamlit App.  
# This includes the sidebar, the title, the footer, and the password check.  
# """  
  
def check_password():  
    """Returns `True` if the user had the correct password."""  
    def password_entered():  
        """Checks whether a password entered by the user is correct."""  
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):  
            st.session_state["password_correct"] = True  
            del st.session_state["password"]  # Don't store the password.  
        else:  
            st.session_state["password_correct"] = False  
    # Return True if the passward is validated.  
    if st.session_state.get("password_correct", False):  
        return True  
    # Show input for password.  
    st.text_input(  
        "Password", type="password", on_change=password_entered, key="password"  
    )  
    if "password_correct" in st.session_state:  
        st.error("😕 Password incorrect")  
    return False


# URLs for SkillsFuture support schemes
URLS = [
    "https://www.skillsfuture.gov.sg/credit/",
    "https://www.skillsfuture.gov.sg/ecg/",
    "https://www.skillsfuture.gov.sg/initiatives/early-career/tesa",
    "https://www.skillsfuture.gov.sg/level-up-programme/",
    "https://www.skillsfuture.gov.sg/careertransition",
    "https://www.skillsfuture.gov.sg/initiatives/students/jobs-skills",
    "https://www.skillsfuture.gov.sg/initiatives/mid-career/midcareersupportpackage",
    "https://www.skillsfuture.gov.sg/mid-career-enhanced-subsidy/",
    "https://www.skillsfuture.gov.sg/initiatives/mid-career/leadershipdevelopment",
    "https://www.skillsfuture.gov.sg/mid-career-training-allowance/",
    "https://www.skillsfuture.gov.sg/jobseeker-support",
    "https://www.skillsfuture.gov.sg/workstudy/wscert",
    "https://www.skillsfuture.gov.sg/workstudy/wsdeg",
    "https://www.skillsfuture.gov.sg/workstudy/wsdip",
    "https://www.skillsfuture.gov.sg/workstudy/wspostdip",
    "https://www.myskillsfuture.gov.sg/content/portal/en/career-resources/career-resources/education-career-personal-development/skillsfuture-career-transition-programme.html",
    "https://www.myskillsfuture.gov.sg/content/portal/en/career-resources/career-resources/education-career-personal-development/SkillsFuture_Level-Up_Programme.html",
    "https://www.myskillsfuture.gov.sg/content/portal/en/career-resources/career-resources/education-career-personal-development/skills-training-advice.html",
    "https://www.myskillsfuture.gov.sg/content/portal/en/career-resourceshttps://www.skillsfuture.gov.sg/credit//career-resources/education-career-personal-development/SkillsFuture_Credit.html",
    "https://www.myskillsfuture.gov.sg/content/portal/en/career-resources/career-resources/education-career-personal-development/workfare_skills_support_scheme_wss.html",
    "https://www.myskillsfuture.gov.sg/content/portal/en/career-resources/career-resources/education-career-personal-development/Courses-to-Support-Caregivers-Training.html",
]