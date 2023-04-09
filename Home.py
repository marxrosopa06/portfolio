import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image

#PAGE CONFIGURATION
st.set_page_config(
     page_title="Marx's Portfolio",
     page_icon="📘",
     initial_sidebar_state="expanded",
     layout="wide",
)

st.header("👋🏽 Hello! Welcome to my analytics portfolio!")
st.header(" ")

col1, col2 = st.columns(2)

with col1:
    st.code("#yesthiscodeblockisintentionalbecauseitlookscool \n"
            "image = Image.open('me_in_Osaka_Japan.jpg') \n"
            "st.image(image)")
    image_me = Image.open("me_in_Osaka_Japan.jpg")
    st.image(image_me)

with col2:
    st.subheader("My name is Marius Rosopa and this portfolio consists of my works, mostly related to Machine Learning and Python coding.")
    st.write(":green[**A little bit about me**]: I'm 25 and from the Philippines 🇵🇭; proud Filipino here!")
    st.write(":green[**More things about me**]: I've been working an analytics professional since I started my career journey. My goal is to get better at my career, learn more \n"
             "technologies and ideas in the field of analytics and Data Science, and eventually become a senior analyst one day.")
    st.write(":green[**A few more things about me**]: I love spicy food, specifically Indian and Thai cuisines. I've always wanted to live and work in Australia -- that place \n"
             "is heavenly! I'm a BIIIIIIG fan of horror movies and true crime documentaries.")
    st.write(":green[**Last few things about me**]: I've worked in 3 companies in the last 5 years, namely Citibank, Thakral One, and Atlassian: \n")
    st.caption(":orange[**Citibank**]: Worked as a Data Analyst in the Supply Chain department. Did supplier analytics by monitoring savings and expenditure of Citi's global suppliers.")
    st.caption(":orange[**Thakral One**]: Held the role of Associate Analytics Consultant and was deployed to work for one of Philippine's biggest telco companies, Globe. \n"
               "Worked as a Digital Analyst in the Data Innovations team. Did campaign and website analytics by analyzing Globe's website statistics.")
    st.caption(":orange[**Atlassian**]: Took the role of Data Analyst. Did Sales Analytics and analyzed statistics of Atlassian's different products and services.")

# st.write("This is hosted through Streamlit (https://streamlit.io/)")