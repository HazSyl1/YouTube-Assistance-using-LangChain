import streamlit as st
import langchain_2 as lch
import textwrap 

background_color = "#17A2B8"  
custom_css = """
    <style>
    body{
        color : #333
    }
    h1{
        color:white ;
        font-style : italic;
        
    }
    
    [data-testid="stSidebarUserContent"]
    {
        background: linear-gradient(to left top, rgb(142,68,92,0.3),rgb(20,29,34,0.5));
    }
    [class="stTextLabelWrapper st-emotion-cache-y4bq5x ewgb6651"]
    {
        background: rgb(21,29,32,0.4);
    }
    
    [class="st-emotion-cache-64tehz e1f1d6gn2"]
    {
        background: linear-gradient(to right bottom,#151f21,#162022);
    }
    
    [class="appview-container st-emotion-cache-1wrcr25 ea3mdgi4"]
{
            background-image: url(https://assets.awwwards.com/awards/images/2019/06/color-UI-cover-big-5.jpg) ;
            background-size: cover;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)



st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url=st.text_area(label="Enter YouTube URL", max_chars=50)
        
        query=st.text_area(label="Ask me about the Video!", max_chars=40, key="query")
        
        submit_button=st.form_submit_button(label="Submit")


if youtube_url and query and submit_button:
    try:
        wait=st.text("Processing Please Wait...")
        db = lch.create_vector_db_from_youtbe_url(youtube_url)
        response = lch.get_response_from_query(db , query)
        wait.empty()
        st.subheader("Answer:")
        st.text(textwrap.fill(response , width=80))
        
    except Exception as e:
        print(e)
        wait.empty()
        st.text("Something went wrong")
        