from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS

from dotenv import  load_dotenv

load_dotenv()
video_url="https://www.youtube.com/watch?v=lG7Uxts9SXs"

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
    
    llm = OpenAI(model=" gpt-3.5-turbo-instruct")
    
    prompt=PromptTemplate(
        input_variables=["question","docs"],
        template="""
        You are a helpful YouTube assistant that can answer questions about videos based on the video's transcript.
        
        Answer the following questions: {question}
        By searcing the following video transcript: {docs}
        
        Only use the factual information from the transcript to asnwer the questions.
        
        If you feel like you dont have enough information to answer the questions , just say "I dont know".set
        
        Your answers should be detailed.  
        """
    )
    
    chain =LLMChain(llm=llm ,prompt=prompt )
    
    response = chain({"question":query,"docs":docs_page_content})
    response=response.replace("\n","")
    return response

print(get_response_from_query(db=create_vector_db_from_youtbe_url(video_url),query='What is the video about?')
      )

    
    