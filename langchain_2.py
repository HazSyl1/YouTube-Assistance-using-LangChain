from langchain_community.document_loaders import YoutubeLoader

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import  load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path("/etc/secrets/.env"), verbose=True, override=True)

output_parser = StrOutputParser()

embeddings = OpenAIEmbeddings()

def create_vector_db_from_youtbe_url(video_url: str)-> FAISS:
    loader=YoutubeLoader.from_youtube_url(video_url)
    transcript=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)
    
    docs=text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs,embeddings)
    return db

def get_response_from_query(db,query,k=4):
    
    docs = db.similarity_search(query,k=k)
    docs_page_content=" ".join([d.page_content for d in docs])
    
    llm = ChatOpenAI(verbose=True)
    
    prompt=ChatPromptTemplate.from_messages(
        ["""
        You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.
        
        Answer the following questions: {question}
        By searcing the following video transcript: {docs}
        
        Only use the factual information from the transcript to asnwer the questions.
        
        If you feel like you dont have enough information to answer the questions , just say "I dont know".set
        
        Your answers should be detailed.  
        """]
    )
    
    chain =prompt | llm | output_parser
    
    response = chain.invoke({"question":query,"docs":docs_page_content})
    response=response.replace("\n","")
    return response

# video_url="https://www.youtube.com/watch?v=lG7Uxts9SXs"

# print(get_response_from_query(db=create_vector_db_from_youtbe_url(video_url),query='What is the video about?')
#       )

    
    
    