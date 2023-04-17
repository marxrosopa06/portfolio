import streamlit as st
from PIL import Image

# APP SET-UPS (PAGE CONFIG, TITLE, SIDEBARS, CONTAINERS)
st.set_page_config(
     page_title="SSAM Data Generator",
     page_icon="🦕",
     initial_sidebar_state="expanded",
     layout="wide",
 )

# LOGO
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://logosandtypes.com/wp-content/uploads/2020/07/atlassian.svg);
                background-size: 200px;
                background-repeat: no-repeat;
                padding-top: 180px;
                background-position: 50px 35px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

# PROPORTIONED SIDEBAR
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 380px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# CENTER DETAILS
st.title("Welcome to the Sales Data Extraction Hub! 🏄‍♂️")
st.write(
    "This tool is developed by the **Sales Analytics - Manila Team** 🇵🇭 for Atlassian internal use."
)

st.markdown("""---""")

st.header(
    "😁 Get to know the app!"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("What is this for?")
    st.markdown(
    "This app is built for Atlassians who wish to retrieve sales data easily."
    )

    st.subheader("What does it do?")
    st.markdown(
    "It constructs a SQL code based on the selections (fields, filters, etc.) you provide, which is then automated, by a push of a button, to query from the Socrates database. \n"
    "It uses the Python framework **databricks-sql-connector** to connect to this database via Databricks. \n"
    "Through this framework, the constructed SQL code is then finally used to return your desired data!"
    )        

with col2:
    st.subheader("When should I use this?")
    st.markdown(
    "If you need quick data pulls for your spend analysis or you want to build a license report for a stakeholder, this is your go-to app! \n"
    )        

    st.subheader("How was this built?")
    st.markdown(
    "This wouldn't be possible without the existence of the Python framework **streamlit**, which is an open-source library generally designed to build web apps. \n"
    "This also uses other packages relevant to Data Analysis and Data Science such as **numpy** and **pandas** for data manipulation and wrangling, **streamlit-aggrid** for the front-end visualization, and more!"
    )  
  

st.markdown("""---""")

st.header(
    "🤷‍♀️ Feeling lost? Confused? Here's a short guide on how to navigate through the app."
)

st.subheader("Overview: The app consists of two main parts - the Sidebar and the Center")

col3, col4 = st.columns(2)

with col3:
    st.write("**Sidebar**: This is where you do your selections such as the fields/columns that you want to be included in your data pull, as well as filters and other advanced options.")

    pic_sidebar = Image.open(r"/Users/mrosopa/Desktop/SSAM_App/media/pic_sidebar.png")
    st.image(pic_sidebar, caption="Sidebar") 

with col4:
    st.write("**Center**: This is where your generated data will appear, in a form of a table. It serves as a preview of your data pull.")

    pic_center = Image.open(r"/Users/mrosopa/Desktop/SSAM_App/media/pic_center.png")
    st.image(pic_center, caption="Center") 

st.subheader("🍻 Sidebar: The life of the party")

st.caption("*Imagine a party. The Sidebar basically represents the attendees. There's no party without people, right?*")

st.write("The components in the Sidebar are the most essential players in this app. \n"
        "They're tasked to build the query for you that is going to be then used to pull your data. \n"
        "(From a technical perspective, this is your SQL code builder.)"
        )

col5, col6 = st.columns(2)

with col5:

    st.write("**Fields**: Select the columns that you want to be included in your data pull.")

    vid_fields = open(r"/Users/mrosopa/Desktop/SSAM_App/media/vid_fields.mov", "rb")   
    vid_fields_read = vid_fields.read()
    st.video(vid_fields_read)

    st.markdown("""---""")

    st.write("**Limit**: Limit the number of datapoints (rows) of your data pull.")

    vid_fields = open(r"/Users/mrosopa/Desktop/SSAM_App/media/vid_limit.mov", "rb")   
    vid_fields_read = vid_fields.read()
    st.video(vid_fields_read)


with col6:
    st.write("**Filters**: Create filters that you want your data pull to be adjusted for.")

    vid_filters = open(r"/Users/mrosopa/Desktop/SSAM_App/media/vid_filters.mov", "rb")
    vid_filters_read = vid_filters.read()
    st.video(vid_filters_read)

    st.markdown("""---""")

    st.write("**Advanced Filters**: Apply additional filters such as Sort and Distinct.")

    vid_filters = open(r"/Users/mrosopa/Desktop/SSAM_App/media/vid_advanced.mov", "rb")
    vid_filters_read = vid_filters.read()
    st.video(vid_filters_read)

    st.caption("**Sort data**: Arranges your data according to your selected field; has the options Descending or Ascending.")
    st.caption("**Show unique values**: Sets the chosen field to only show unique row combinations.")
  
st.subheader("🤡 Filters: The tricky one")

st.caption("*Imagine a prankster in a party. This is it.*")

st.write("The Filters section is the app's core because this is what streamlines your data pull depending on your needs. \n"
        "It relies on the data type of the fields chosen. The interface changes depending on field's data type. \n"
        )

col7, col8, col9, col10 = st.columns(4)

with col7:
    st.write("**String**: Character variables")
    pic_string = Image.open(r"/Users/mrosopa/Desktop/SSAM_App/media/pic_string.png")
    st.image(pic_string, caption="Methods: Text Input (Manual Input) and File Upload (Reads data in a file)")     

with col8:
    st.write("**Date**: Variables in date format")
    pic_date = Image.open(r"/Users/mrosopa/Desktop/SSAM_App/media/pic_date.png") 
    st.image(pic_date, caption="Methods: Range (In between two dates) and Single Date")     

with col9:
    st.write("**Double**: Variables in number format")
    pic_doub = Image.open(r"/Users/mrosopa/Desktop/SSAM_App/media/pic_doub.png")
    st.image(pic_doub, caption="Methods: Range (In between two amounts) and Single Amount")    

with col10:
    st.write("**Boolean**: Conditional variables")
    pic_bool = Image.open(r"/Users/mrosopa/Desktop/SSAM_App/media/pic_bool.png") 
    st.image(pic_bool, caption="Common examples are True/False, 0/1")   

st.subheader("🎉 Center: The main event")

st.caption("*This is the party proper. Here's where we all celebrate!*")

st.write("The Center section is where your data pull will appear, in a form of an **AgGrid table**. \n"
        "It cleanly visualizes and previews your generated data and is highly interactive! \n"
        )

col11, col12 = st.columns(2)

with col11:
    vid_filters = open(r"/Users/mrosopa/Desktop/SSAM_App/media/vid_advanced.mov", "rb")
    vid_filters_read = vid_filters.read()
    st.video(vid_filters_read)
