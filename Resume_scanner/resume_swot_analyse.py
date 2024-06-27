import os
from crewai import Crew, Process
from dotenv import load_dotenv, find_dotenv # Groq, Serper
from langchain_groq import ChatGroq # Mixtral
from utils import *
from agents import agents
from tasks import tasks
load_dotenv(find_dotenv())
import streamlit as st
import json

GROQ_API_KEY = "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"
SERPER_API_KEY = "dc379380cfbe9199ab2358c4e44234150de5ea0c"

# Configuration
os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

#Streamlit
st.title('HR DASHBOARD')

JD_uploaded_files = st.file_uploader("Choose Job Description file")

resume = st.file_uploader("Choose Resume file", accept_multiple_files=True)

# Load the llm
llm = ChatGroq(model="llama3-8b-8192", temperature=0)
#llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# Provided the inputs
#resume = read_all_pdf_pages("data/Sai_Nithin_Resume.pdf")
#job_discription = input("Enter Job Discribtion : ")
job_discription = "Sai"

# Creating agents and tasks
resume_swot_analyser = agents(llm)
# job_requirements_researcher, resume_swot_analyser = agents(llm)

resume_swot_analysis = tasks(llm, job_discription, resume)
#research, resume_swot_analysis = tasks(llm, job_desire, resume)

# Building crew and kicking it off
crew = Crew(
    agents=[resume_swot_analyser],
    tasks=[resume_swot_analysis],
    verbose=1,
    process=Process.sequential
)

# result = crew.kickoff()
# print(result)


if st.button("Submit", type="primary"):
    result = crew.kickoff()
    #print(result)
    json_string = json.dumps(result)
    json_data = json.loads(json_string)
    st.json(json_data)



