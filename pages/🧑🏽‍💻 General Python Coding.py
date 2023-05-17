import math
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

#PAGE CONFIGURATION
st.set_page_config(
     page_title="Marx's Portfolio",
     page_icon="📘",
     initial_sidebar_state="expanded",
     layout="wide")

with st.sidebar:
    selection = st.selectbox("Type of Algorithm:", ("(Intro)", "Data Collator", "Data Hub"))

#INTRO
if selection == "(Intro)":
    st.header("🧑🏽‍💻 Welcome to my :orange[**General Python Coding**] page!")
    st.subheader("This page is dedicated to my works, mostly in my past job in :green[**Atlassian**], because that's where I only was able to showcase my coding skills. \n"
             "This includes 2 of my works: :orange[**Data Collator**] and the :orange[**Data Hub**]")
    
elif selection == "Data Collator":
    st.header("🗳 Data Collator")
    st.subheader("Skills involved: Data Wrangling and Data Manipulation with Pandas")
    st.write(":green[**Problem:**] One of my very first jobs as a new hire is to help a senior analyst with her week-on-week forecasting exercise. The way things are done \n"
             "is by downloading :orange[**17 files in Clari**] (a revenue and forecasting platform), open them one-by-one, filter specific values in several columns, \n"
             "copy, and eventually paste them to a master file.")
    st.write(":green[**Solution:**] I realized that this is such a lengthy (and a tiresome) process, so I created a script that would reduce the time of my doing of this \n"
             "exercise, which usually takes me 10-15 minutes, :orange[**into 3 seconds, with just one click of a script**].")
    
    st.write(":green[**Process:**] \n"
             " \n")
    st.write("1) Set up all the essential packages, needed folders and directories, and the mappings, which were specificed by the senior analyst (for this example, \n"
             "I altered the real mappings to ensure privacy)")
    st.code("import os \n"
            "import glob \n"
            "import pandas as pd \n"
            " \n"
            "#Change every week and quarter \n"
            "main_folder = '/Users/mariusrosopa/Desktop/WoW Forecast/' \n"
            "current_week = 4 \n"
            "current_quarter_year = 'Q2 2023' \n"
            "current_week_folder = 'W' + str(current_week) + ' ' + '('' + current_quarter_year + ')/'' \n"
            "path = main_folder + current_week_folder \n"
            " \n"
            "#File mappings \n"
            "mapping = { \n"
            "   'product1_abc': 'product1_abc_Marius', \n"
            "   'product2_xyz': 'product2_xyz_Marius', \n"
            "   'product3_iop': 'product3_iop_Marius', \n"
            "   'product4_jkl': 'product4_jkl_Lee', \n"
            "   'product5_bnm': 'product5_bnm_Lee', \n"
            "   'product6_tyu': 'product6_tyu_Lee', \n"
            "}")
    
    st.write("2) Now we read each CSV file in the current directory (which is the 'path' variable) and add a new column per file and combine all of their contents \n"
             "into one master file. The purpose of the added column, which contains the filename, is to know from which file did the data come from")
    st.code("#Finds all the CSV files in the current folder \n"
            "files = glob.glob(path + '/*.csv') \n"
            " \n"
            "#Adds a filename column per file and combines all files into a single file \n"
            "empty_list = [] \n"
            " \n"
            "for file in files: \n"
            "   frame = pd.read_csv(file) \n"
            "   frame['filename_col'] = os.path.basename(file) \n"
            "   empty_list.append(frame) \n"
            " \n"
            "raw_df = pd.concat(empty_list) \n")
    
    st.write("3) Now we loop through our mapping dictionary using the file name column that we built in the previous step. This is because during that exercise, \n"
             "the file names produced by Clari were very lengthy and confusing; it contains the name of the specific product (in our mapping list, they're 'product 1', \n"
             "'product 2', 'product 3', etc.) plus a bunch of other weird texts. The purpose of this is to find the specific product name in the lengthy file name \n"
             "and use those to lookup in the dictionary to return the same product name but with the added director name (hence the names 'Marius' and 'Lee' but I had to \n"
             "alter those to my personal names to ensure privacy)")
    st.code("#Loops through the mapping dict and maps each filename accordingly \n"
            "final_mapping = {} \n"
            "filename_unique_list = raw_df['filename_col'].unique().tolist() \n"
            " \n"
            "for map in list(mapping.items()): \n"
            "   for filename in filename_unique_list: \n"
            "       if map[0] in filename: \n"
            "           final_mapping[filename] = map[1] \n")
    st.caption("**Example: Product name is :orange[**'product1abc'**]. It's file name is :orange[**'EXPORT_2023_FORECAST_product1abc_Dec01-Dec312023_includes_all_columns.csv'**]. The script \n"
               "looks for the product name in the file name and once found, it will append the product name in the newly added file name column in the master file. This is to \n"
               "to know from which file did the specific data come from**")
    
    st.write("4) Now we create a :green[**raw_mapping**] column that gets the product name with the director name from the file name and then split those values to separate the \n"
             "product name and the director name in two different columns, hence the :green[**mapping**] and :green[**name_org**] columns")
    st.code("#Create new columns and clean the data \n"
            "raw_df['raw_mapping'] = [final_mapping[key] for key in raw_df['filename_col']] \n"
            " \n"
            "raw_df['mapping'] = [key.split('_',1)[0] for key in raw_df['raw_mapping']] \n"
            "raw_df['name_org'] = [key.split('_',1)[1] for key in raw_df['raw_mapping']]")
    
    st.write("5) We now then filter the columns with the needed parameters, do some final cleaning, and finally export to a file")
    st.code("#Manipulate the data using the filters specified \n"
            "#Data value = Excluding Blanks \n"
            "dfs_cleaned_1 = raw_df.dropna(subset=['Data Value']) \n"
            " \n"
            "#Week = current week \n"
            "dfs_cleaned_2 = dfs_cleaned_1.loc[dfs_cleaned_1['Week']==current_week] \n"
            " \n"
            "#Data Type = Closed, Forecast Value \n"
            "dfs_cleaned_3 = dfs_cleaned_2.loc[(dfs_cleaned_2['Data Type']=='type1') | (dfs_cleaned_2['Data Type']=='type5')] \n"
            " \n"
            "#Field = Exp/Upgrade Outlook, New Outlook \n"
            "final_df = dfs_cleaned_3.loc[(dfs_cleaned_3['Field']=='field2') | (dfs_cleaned_3['Field']=='field4')] \n"
            " \n"
            "#Final data clean \n"
            "final_df.reset_index(drop=True, inplace=True) \n"
            "final_df.index += 1 \n"
            "final_df.index.name = 'row_num' \n"
            " \n"
            "#Export final dataset \n"
            "final_df.to_csv(path + str(current_quarter_year) + '_' + 'W' + str(current_week) + '_' + 'final_data.csv')")
    
    st.write(" \n")
    st.subheader("Here's a video showing how the whole process works")
    video_data_collator = open('data_collator.mp4', 'rb')
    video_bytes_data_collator = video_data_collator.read()
    
    st.video(video_bytes_data_collator)
     
    st.caption(":red[**Note: Actual names, data types, fields, had to be altered for privacy reasons. Values are simply generic.**]")



elif selection == "Data Hub":
    st.header("🖥 Data Hub (a web application via Streamlit)")
    st.subheader("Skills involved: Data Wrangling and Data Manipulation with Pandas and Numpy, Web App Development with Streamlit")
    st.write("Back in :green[**Atlassian**], one of my main tasks is extracting data reports for internal stakeholders. These reports serve as their data-driven efforts \n"
             "to assist their decision-making related to Atlassian's sales and products. We normally do the extraction through Databricks using SQL.")
    st.write("This :orange[**sparked an idea and became my biggest passion project as an analytics professional to date**]. I realized that I could fully automate this process: \n"
             "where I could create :green[**a web platform that internal stakeholders could simply go to and pull reports themselves**], without having to go through Jira, file a ticket \n"
             "for our team, and wait for their reports to be sent (takes days, depending on the priority and team's availability).")
    st.write("I was able to :green[**develop a fully-working web app**] and even got the chance to present this to my higher-ups. Unfortunately, it didn't progress to final stage \n"
             "(approval and actual utilization) 😢")
    st.subheader("Video of how this app works is available upon request.")
