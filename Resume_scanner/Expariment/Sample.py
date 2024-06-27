import tiktoken
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import chromadb
from langchain.chains import RetrievalQA
from datetime import datetime
from langchain_openai.chat_models import ChatOpenAI
import pprint
from langchain_openai import OpenAIEmbeddings


import os

os.environ["HUGGINGFACEHUB_API_TOKEN"]="hf_token"
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
class Document:
    def __init__(self, text, metadata={}):
        self.page_content = text
        self.metadata = metadata


def load_file(filepath):
    
        if filepath.endswith(".docx"):
            loader = UnstructuredFileLoader(filepath)
            docs=loader.load()
            return docs
        elif filepath.endswith(".pdf"):
            loader=UnstructuredPDFLoader(filepath)
            docs=loader.load()
            return docs



def extract_information(docs):
    template = """\
    For the following text, extract the following information:
    Skills: what are the technical and non technical skills? \
    Answer output them as a comma separated Python list.

    Education: What is the highest education of the candidate and what is the GPA as mentioned in the text?\
    Answer Output should be the university/college name and GPA if given in text, output them as a comma separated Python list.

    Projects: Extract all project titles mentioned in a text\
    and output them as a comma separated Python list.

    Publications: Extract all publication titles mentioned in a text\
    and output them as a comma separated Python list.

    Work experience: Extract all organisation name where he/she has worked along with number of years or months worked there and also extract designation\
    and output them as a comma separated Python list.

    Format the output as JSON with the following keys:
    Skills
    Education
    Projects
    Publications
    Work experience

    text: {text}
    """
    prompt_template = PromptTemplate(input_variables=["text"],template=template)
    # print(prompt_template)
    # repo_id="mistralai/Mistral-7B-Instruct-v0.3"
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct"
    llm=HuggingFaceEndpoint(repo_id=repo_id,temperature=0.7)
    chain = prompt_template | llm
    output = chain.invoke(docs[0].page_content)
    print(output)
    return output

def vectorstore(documents,jd):
    embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en",
                                        model_kwargs={'device': 'cpu','trust_remote_code':True})
    #embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d%H%M%S")
    print("Current Time:", formatted_time)
    collection_name=formatted_time #need to give a unique name in the future this is done to avoid the retrieval of previous instance pdfs
    vectordb = Chroma.from_documents(documents, embeddings,collection_name=collection_name)
    endtime = datetime.now()
    end_time = endtime.strftime("%Y-%m-%d-%H:%M:%S")
    print("Current Time:", end_time)

    #repo_id="meta-llama/Meta-Llama-3-8B-Instruct"
    repo_id="mistralai/Mistral-7B-Instruct-v0.3"
    llm=HuggingFaceEndpoint(repo_id=repo_id,temperature=1)
    # llm = ChatOpenAI(temperature=0.7)
    # warning = "If you don't know the answer, just say that you don't know, don't try to make up an answer"
    job_description = jd
    question ="You are an expert in talent acquisition that helps determine the best candidate among multiple suitable resumes.Use the following pieces of context to determine the best resume given a job description.Job description:\n"+job_description+"\nBased on the given job description"
    query = question + "give an overall score for the resumes which is good fit to the job based on skills,education and work experience mentioned in it?"
    # query = question + "short list resumes which is good fit to the job based on skills,education and work experience mentioned in it?"+warning
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs_retrieved=retriever.invoke(query)
    # qa_chain = RetrievalQA.from_chain_type(llm=llm,  
    #                               retriever=retriever, 
    #                               return_source_documents=True
    #                               )
    # result=qa_chain.invoke(query)
    # print(result["result"])
    for doc in docs_retrieved:
        print(doc.metadata)
    # print(result["source_documents"][0].metadata)


def main():
    resumes=[#"/Users/kranthivardhankurumindla/Documents/HR_usecase/Aajin Roy.pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Adarsh.pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Akhil_Rajan_Resume.pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/midhunresume.pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_ArvindKumarJangid[12y_0m].pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_ChandrajeetPratapSingh[5y_0m].pdf",
            "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_DEEPAMERLINDIXONK[4y_3m].pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_JeevanDhadge[7y_2m].pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_KUNALTAJANE[5y_1m].pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_ParagKumarJain[5y_9m].pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_RamandeepBains[5y_1m].pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_RohitKumarGupta[8y_0m].pdf",
            #  "/Users/kranthivardhankurumindla/Documents/HR_usecase/Naukri_ShephaliGupta[7y_6m].pdf"
            ]
    documents=[]
    for resume in resumes:
        docs=load_file(resume)
        print(docs)
    #     extracted_information=extract_information(docs)
    #     document = Document(extracted_information, metadata={"filename":os.path.basename(resume)})
    #     documents.append(document)
    jd=load_file("/Users/kranthivardhankurumindla/Documents/HR_usecase/Full-stack Data Scientist_Job Description_v2.docx")
    # extract_information(docs)
    # vectorstore(documents,jd[0].page_content)

if __name__ == "__main__":
    main()  


# example of information extraction done for the Aajin Roy resume
# {
#     "Skills": ["Machine Learning", "Logistic Regression", "KNN", "CART", "Random Forest", "AdaBoost", "Gradient Boost", "XG Boost", "Linear Regression", "K- Means", "PCA", "LDA", "Text analytics/NLP", "CNN", "Recurrent Neural Network", "ANN", "TF-IDF", "Sentiment Analytics", "Time Series", "Tableau", "PowerBI", "Seaborn", "ggplot2", "Matplotlib", "Descriptive statistics", "Inferential Statistics", "Prescriptive Statistics", "Python", "PowerShell", "PyTorch", "PySpark", "SnowFlake", "JAVA", "MongoDB", "SQL Server", "Big Data", "Azure Databricks", "Delta Lake", "Azure Data Factory"],
#     "Education": ["B.tech in Electronics and Communication, Adi Shankara Institute of Engineering and Technology (KTU)", "7.7"],
#     "Projects": ["Recommendation System", "Network Traffic Analysis", "Reactive and predictive churn of customers", "Credit Card Fraud Detection", "Formula1 Racing Real-world project using PySpark and SQL", "Tracking system using Facial recognition"],
#     "Publications": [],
#     "Work experience": ["Data Scientist, TCS", "Kochi", "Sep 2020", "Aajin Roy", "", "", "Machine Learning Engineer, Curvelogics Advanced Technology Solutions Pvt Ltd", "Kerala", "2021-12", "Adarsh Reghuvaran", "Assistant Data Scientist, Curvelogics Advanced Technology Solutions Pvt Ltd", "Kerala", "2020-09", "Adarsh Reghuvaran", "Associate Analytics Consultant, Exponential Digital Solutions", "Kerala", "2018-11", "Adarsh Reghuvaran"]
#     }


# Adarsh.pdf
# {
#     "Skills": ["Machine Learning", "Deep Learning", "Pytorch", "TensorFlow", "SQL", "Computer Vision", "NLP", "Python", "C++", "PowerBl", "Tableau", "OpenCv", "Numpy", "Pandas"],
#     "Education": ["University of Kerala - Kerala, India (GPA: 8.8/10)", "Scaler Neovarsity - India"],
#     "Projects": ["Deep Learning Model for Image Classification", "Object Detection and Tracking", "Smart City Traffic Management System", "Predictive Maintenance System for Industrial Machines", "Natural Language Processing for Text Classification"],
#     "Publications": ["Deep Learning for Image Classification: A Comprehensive Review", "Predictive Maintenance System for Industrial Machines: A Deep Learning Approach"],
#     "Work experience": ["Trenser Technology Solutions (P) Ltd, Kerala (2021-12 - Machine Learning Engineer)", "Curvelogics Advanced Technology Solutions Pvt Ltd, Kerala (2020-09 - Assistant Data Scientist)", "Exponential Digital Solutions, Kerala (2018-11 - Associate Analytics Consultant)"]
# }