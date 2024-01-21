import streamlit as st
import langchain_2 as lch
import textwrap 

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url=st.text_area(label="Enter YouTube URL", max_chars=50)
        
        query=st.text_area(label="Ask me about the Video!", max_chars=40, key="query")
        
        submit_button=st.form_submit_button(label="Submit")


if youtube_url and query:
    db = lch.create_vector_db_from_youtbe_url(youtube_url)
    response = lch.get_response_from_query(db , query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response , width=80))
    
    