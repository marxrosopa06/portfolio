import streamlit as st
import pandas as pd
import numpy as np
import time
from databricks import sql
from st_aggrid import AgGrid
import pandas as pd
import numpy as np

# STATIC LIST OF SCHEMAS AND TABLES (UPDATE IF YOU WISH TO ADD SCHEMAS AND TABLES)
schema_table_list = pd.read_csv("/Users/mrosopa/Desktop/SSAM_App/schema_table_list.csv")

# # FUNCTION TO GET THE LIST OF FIELDS AND DATA TYPES (NECESSARY TO RUN IF YOU UPDATE SCHEMA TABLE LIST)
# def final_list():

#     i = 0
#     dfs = []

#     for i in range(0,len(schema_table_list)):
#         schema = schema_table_list.iloc[i]["schema"]
#         table = schema_table_list.iloc[i]["table"]

#         cursor.execute(f"DESCRIBE {schema}.{table}")
#         result = cursor.fetchall()
#         result_df = pd.DataFrame(result)
#         result_df["schema"] = schema
#         result_df["table"] = table
    
#         dfs.append(result_df)
#         combined = pd.concat(dfs)

#     combined_cleaned = combined.rename(columns={0:"field", 1:"data_type", 2:"comment"}).drop("comment",1)
#     df = combined_cleaned[["schema","table","field","data_type"]]
#     df["concat"] = df["schema"] + df["table"] + df["field"]

#     return df

# final_list().to_csv("/Users/mrosopa/Desktop/SSAM_App/final_list.csv", index=False) # Change the directory if necessary

# FINAL LIST OF SCHEMAS, TABLES, FIELDS, DATA TYPES
df = pd.read_csv("/Users/mrosopa/Desktop/SSAM_App/final_list.csv")

# LOGO
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://logosandtypes.com/wp-content/uploads/2020/07/atlassian.svg);
                background-size: 200px;
                background-repeat: no-repeat;
                padding-top: 170px;
                background-position: 80px 40px;
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

# APP SET-UPS (PAGE CONFIG, TITLE, SIDEBARS, CONTAINERS)
sidebar_p1 = st.sidebar
sidebar_p2 = st.sidebar
sidebar_p3 = st.sidebar
sidebar_p4 = st.sidebar
sidebar_p5 = st.sidebar
center_login = st.empty()
center_main = st.container()
center_p1 = st.container()
center_p2 = st.container()
center_p3 = st.container()

# LOG-IN WINDOW PANE
col1, col2 = st.columns(2)

if "login_submit" not in st.session_state:
    st.session_state["login_submit"] = False

if st.session_state["login_submit"] == False:

    with center_login.container():

        st.title("Login - Sales Data Extraction Hub 🏄‍♂️")
        st.header("Want to use the tool? Before anything else, let us know you first!")

        req_name = st.text_input("Requester Name:")
        req_name_empty = st.container()

        req_team = st.text_input("Team/Department:")
        req_team_empty = st.container()

        req_reason = st.text_input("Reason for data extraction:")
        req_reason_empty = st.container()

        req_loc = st.selectbox("Location:", ("AMER", "APAC", "EMEA"))

        help = "These are the common use cases and some of the usual data that you'll find for each report type:\n"
        help_l = "**License**: If you need data for licenses/instances of a particular domain or product, along with the number of users. *Common fields: SEN, Unit Count, Purchase and Expiry Dates.* \n "
        help_s = "**Spend**: If you need data for licenses/instances, along with the price paid by the customer. *Common fields: SEN, Product, Customer Paid Price.* \n"

        req_type = st.selectbox("Report type:", ("License", "Spend"), help=help+"\n"+help_l+"\n"+help_s)
        st.session_state.req_type = req_type

        req_comms = st.text_area("Additional comments:")

        req_submit = st.button("Submit")

        if req_submit:

            if req_name == "" or req_team == "" or req_reason == "":

                if req_name == "":
                    with req_name_empty:
                        st.warning("⬆ Oops! You forgot this fill this one out!")

                if req_team == "":
                    with req_team_empty:
                        st.warning("⬆ Oops! You forgot this fill this one out!")

                if req_reason == "":
                    with req_reason_empty:
                        st.warning("⬆ Oops! You forgot this fill this one out!")
                
            elif req_name != "" and req_team != "" and req_reason != "":

                st.session_state["login_submit"] = True

                with st.spinner("Loading..."):
                    time.sleep(2.5)

## MAIN PAGE (AFTER TRIGGERING SUBMIT BUTTON IN LOG-IN WINDOW)        
if st.session_state["login_submit"] == True:

    center_login.empty()

    # FUNCTION: DATABRICKS SQL REQUIREMENTS
    @st.cache(allow_output_mutation=True, show_spinner=False)
    def get_connection():

        connection = sql.connect(
        server_hostname = "atlassian.cloud.databricks.com",
        http_path = "sql/protocolv1/o/0/1017-061509-vexed564",
        access_token = "dapi49a858e2d79d5569ac3eedd6740c5809"
        )
        
        cursor = connection.cursor()
        
        return cursor

    # FUNCTION: SQL INITIAL CODE
    @st.cache(allow_output_mutation=True, show_spinner=False)
    def get_sql():
        return get_connection().execute("SET spark.databricks.credentials.assumed.role=arn:aws:iam::897486716820:role/prod-databricks-data-access-insider")

    # TOP SIDEBAR: FIELD DROPDOWNS
    with sidebar_p1:

        # Info
        st.warning("**Note**: The table saves your last data pull state. Leaving the fields list empty will make it disappear. Preserve by keeping at least one field included.")

        # Pre-select Checkbox
        st.caption(f"Report type: {st.session_state.req_type}")
        preselected_fields = st.checkbox(f"Preselect fields")

        # Title and Header
        st.header("Fields")

        # Initialize Session States
        if "data" not in st.session_state:
            st.session_state.data = []

        if "field_choice" not in st.session_state:
            st.session_state.field_choice = []

        # Schemas and Tables (Defaulted to "atlas" and "sale_insider" for now)
        schema_choice = "atlas"
        table_choice = "sale_insider"

        # Multi-select: Fields 
        field_list = df.loc[(df["schema"]==schema_choice) & (df["table"]==table_choice)]["field"].tolist()

        if preselected_fields and st.session_state.req_type == "License":
            field_choice = st.multiselect("Select fields", field_list, default=["sen", "account_email_domain", "unit_count", "product", 
            "partner_name", "entitlement_level", "platform", "maintenance_start_date", "maintenance_end_date"])

        elif preselected_fields and st.session_state.req_type == "Spend":
            field_choice = st.multiselect("Select fields", field_list, default=["sen", "account_email_domain", "unit_count", "product", 
            "list_price_amount", "partner_name", "entitlement_level", "platform", "maintenance_start_date", "maintenance_end_date"])

        else:
            field_choice = st.multiselect("Select fields", field_list)

    # BOTTOM SIDEBAR: FILTER DROPDOWNS
    with sidebar_p2:

        # Title
        st.header("Filters")

        # Multi-select: Filters
        filter_choice = st.multiselect("Select fields that you want to use as filters", field_list)
        data_type_list = df["data_type"].unique().tolist()
        filter_choice_specific = []

        # Expanders: Filters
        for filter in filter_choice:

            data_type_identifier = df[df["concat"]==f"{schema_choice}{table_choice}{filter}"]["data_type"]
            
            with st.expander(f"{filter}"):

                # Filter type: Double
                if (data_type_identifier == "double").bool():
                    double_indicator = st.radio("Method:", ("Range", "Single amount"), key=f"{filter}")
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
                    
                    if double_indicator == "Range":
                        double_lower = st.text_input("From", key=f"{filter}")
                        double_upper = st.text_input("To", key=f"{filter}")
                        doub_ftr_value = f"({filter} >= '{double_lower}' AND {filter} <= '{double_upper}')"
                        st.warning(f"❗️For large values, commas are not needed.")

                    elif double_indicator == "Single amount":
                        double_date_method = st.radio("Condition:", ("Less than", "Greater than", "Less than or Equal to", "Greater than or Equal to", "Equal to", "Not equal to"))
                        double_value = st.text_input("Amount", key=f"{filter}")
                        st.warning(f"❗️For large values, commas are not needed.")

                        if double_date_method == "Less than":
                            doub_ftr_value = f"{filter} < '{double_value}'"
                        elif double_date_method == "Greater than":
                            doub_ftr_value = f"{filter} > '{double_value}'"
                        elif double_date_method == "Less than or Equal to":
                            doub_ftr_value = f"{filter} <= '{double_value}'"
                        elif double_date_method == "Greater than or Equal to":
                            doub_ftr_value = f"{filter} >= '{double_value}'"
                        elif double_date_method == "Equal to":
                            doub_ftr_value = f"{filter} = '{double_value}'"
                        elif double_date_method == "Not equal to":
                            doub_ftr_value = f"{filter} <> '{double_value}'"

                    try:
                        filter_choice_specific.append(doub_ftr_value)
                    except NameError:
                        pass

                # Filter type: Boolean
                elif (data_type_identifier == "boolean").bool():
                    bool_value = st.radio("Condition:", ("True", "False"), key=f"{filter}")
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

                    if bool_value == "True":
                        bool_ftr_value = f"{filter} = 'True'"
                    
                    elif bool_value == "False":
                        bool_ftr_value = f"{filter} = 'False'"
                    
                    try:
                        filter_choice_specific.append(bool_ftr_value)
                    except NameError:
                        pass

                # Filter type: Date
                elif (data_type_identifier == "date").bool():
                    date_indicator = st.radio("Method:", ("Range", "Single date"), key=f"{filter}")
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

                    if date_indicator == "Range":
                        date_lower = st.text_input("From: (Format: YYYY-MM-DD)", key=f"{filter}")
                        date_upper = st.text_input("To: (Format: YYYY-MM-DD)", key=f"{filter}")
                        date_ftr_value = f"({filter} >= '{date_lower}' AND {filter} <= '{date_upper}')"

                    elif date_indicator == "Single date":
                        single_date_method = st.radio("Condition:", ("Before", "After", "Before (Including)", "After (Including)", "Equal to", "Not equal to"))
                        date_value = st.text_input("Date (Format: YYYY-MM-DD)", key=f"{filter}")

                        if single_date_method == "Before":
                            date_ftr_value = f"{filter} < '{date_value}'"
                        elif single_date_method == "After":
                            date_ftr_value = f"{filter} > '{date_value}'"
                        elif single_date_method == "Before (Including)":
                            date_ftr_value = f"{filter} <= '{date_value}'"
                        elif single_date_method == "After (Including)":
                            date_ftr_value = f"{filter} >= '{date_value}'"
                        elif single_date_method == "Equal to":
                            date_ftr_value = f"{filter} = '{date_value}'"
                        elif single_date_method == "Not equal to":
                            date_ftr_value = f"{filter} <> '{date_value}'"
                    
                    try:
                        filter_choice_specific.append(date_ftr_value)
                    except NameError:
                        pass

                # Filter type: String, Int, BigInt
                elif (data_type_identifier == "string").bool() or (data_type_identifier == "int").bool() or (data_type_identifier == "bigint").bool():
                    str_indicator = st.radio("Method", ("Text input", "File upload"), key=f"{filter}")
                    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

                    if str_indicator == "Text input":
                        str_value = st.text_input("Enter details", key=f"{filter}")
                        st.info("🙋 Best for **a couple of values**. For multiple entries, **separate your values with a space and a comma** (i.e. gmail.com, gmail.co).")
                        st.warning(f"❗️If the table returned empty, it means that there's no available data for the **{filter}/s** you provided. *Sometimes it's just the spelling, you know?* 😉")

                        str_value_tolist = str_value.split(" ")

                        if len(str_value_tolist) == 1 or len(set([word.find(",") for word in str_value_tolist])) == 1:
                            str_dum_value = " ".join(str_value_tolist)
                            str_ftr_value = f"{filter} = '{str_dum_value}'"

                        elif len(str_value_tolist) > 1:
                            str_value_tolist_more = str_value.split(", ")
                            str_value_tolist_more_cleaned = ", ".join("'" + item + "'" for item in str_value_tolist_more)
                            str_ftr_value = f"{filter} IN ({str_value_tolist_more_cleaned})"

                        try:
                            filter_choice_specific.append(str_ftr_value)
                        except NameError:
                            pass

                    elif str_indicator == "File upload":
                        str_uploaded_file = st.file_uploader("Upload file", key=f"{filter}")

                        if str_uploaded_file is not None:
                            str_uploaded_file_df = pd.read_csv(str_uploaded_file, header=None)

                            str_uploaded_file_df_tolist = list(str_uploaded_file_df.iloc[:, 0])
                            st.write(f"Count of **{filter}s**: {len(str_uploaded_file_df_tolist)}")

                            str_ftr_value_write = ", ".join(str_uploaded_file_df_tolist)
                            st.write(f"**Values**: {str_ftr_value_write}")

                            str_ftr_value_cleaned = ", ".join("'" + item + "'" for item in str_uploaded_file_df_tolist)
                            str_ftr_value = f"{filter} IN ({str_ftr_value_cleaned})"
                            
                            try:
                                filter_choice_specific.append(str_ftr_value)
                            except NameError:
                                pass
                        
                        st.info("🗣 Best for **a ton amount of values**.")
                        st.warning(f"❗️File upload accepts **CSV and XLSX files only**. Make sure your list of **{filter}s** are placed at the upper-leftmost part (cell A1) of the file with no header.")

    with sidebar_p3:

        # Title and Header
        st.header("Limit")

        # Text Input: Limit
        limit_check = st.text_input("Limit data points (defaulted to 1000)", key="limit", value=1000)

        def limit_add():
            if limit_check:
                limit_value = f"LIMIT {limit_check}"

            else:
                limit_value = " "
            
            return limit_value

    with sidebar_p4:

        with st.expander("Advanced"):

            # Checkbox: Order
            order_check = st.checkbox("Sort data")

            if order_check:
                order_choice = st.multiselect("Select a field", field_list)
                order_indicator = st.radio("Sort method:", ("Descending", "Ascending"), key="order")
                st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

                def order_add():
                    if order_choice != []:
                        order_list_cleaned = (", ".join(order_choice))

                        if order_indicator == "Ascending":
                            order_value = f"ORDER BY {order_list_cleaned} ASC"
                        elif order_indicator == "Descending":
                            order_value = f"ORDER BY {order_list_cleaned} DESC"               

                    else:
                        order_value = " "

                    return order_value
            else:
                def order_add():
                    return " "

            # Checkbox: Distinct
            distinct_check = st.checkbox("Show unique values")
            field_choice_with_distinct = []

            if distinct_check:
                distinct_choice = st.selectbox("Select a field", field_choice)

                distinct_holder_idx = field_choice.index(distinct_choice)
                distinct_holder = "DISTINCT " + field_choice[distinct_holder_idx]
                field_choice_with_distinct.insert(0, distinct_holder)

                for field in field_choice:
                    if field == distinct_choice:
                        pass
                    
                    else:
                        field_choice_with_distinct.append(field)

                field_choice = field_choice_with_distinct
                
            else:
                pass

            # Apply
            apply_advanced = st.button("Apply")

    # FUNCTION: MAIN DATA PULL
    @st.experimental_memo(show_spinner=False)
    def run_sql(f, schema, table, order, limit):

        with st.spinner("Running now, be patient okay?"):

            # Run SQL intial code to query from Databricks
            get_sql()

            field_list_cleaned = (", ".join(f))

            get_connection().execute(f"SELECT {field_list_cleaned} FROM {schema}.{table} {order} {limit}")
            sql_run = get_connection().fetchall()
            sql_df = pd.DataFrame(sql_run, columns=field_list_cleaned.split(", "))
            sql_df = sql_df.reset_index()
            sql_df["index"] = np.arange(1, len(sql_df) + 1)
            sql_df.rename(columns = {"index":"#"}, inplace=True)

            return sql_df

    @st.experimental_memo(show_spinner=False)
    def run_sql_with_where(f, schema, table, filter, order, limit):

        with st.spinner("Running now, be patient okay?"):

            # Run SQL intial code to query from Databricks
            get_sql()

            field_list_cleaned = (", ".join(f))

            get_connection().execute(f"SELECT {field_list_cleaned} FROM {schema}.{table} WHERE {filter} {order} {limit}")
            sql_run = get_connection().fetchall()
            sql_df = pd.DataFrame(sql_run, columns=field_list_cleaned.split(", "))
            sql_df = sql_df.reset_index()
            sql_df["index"] = np.arange(1, len(sql_df) + 1)
            sql_df.rename(columns = {"index":"#"}, inplace=True)

            return sql_df

    with sidebar_p5:

        run_script = st.button("Run")

    # CENTER: DATAFRAME
    with center_p1:

        i = 0

    with center_p2:

        # Case 1: Selecting fields with no filters
        if field_choice != [] and filter_choice == []:
            field_list_cleaned = (", ".join(field_choice))

            # Performs the data pull when any of the buttons are clicked (run script and apply advanced)
            if run_script or ((order_check or distinct_check) and apply_advanced):
                st.session_state.data = run_sql(field_choice, schema_choice, table_choice, order_add(), limit_add())
                AgGrid(st.session_state.data)
                st.session_state.field_choice = field_choice
                i += 1

            # Preserves the previously ran data pull should there be any changes in the fields
            elif not run_script and st.session_state.field_choice != []:
                AgGrid(st.session_state.data)
                i += 1

        # Case 2: Selecting fields with filters
        elif field_choice != [] and filter_choice != []:
            field_list_cleaned = (", ".join(field_choice))
            
            # Joins the filters altogether with AND if more than 1 is selected
            filter_choice_specific_cleaned = " AND ".join(filter_choice_specific)

            # If the user decides to run the script again without specifying a filter (proofing)
            if run_script and filter_choice_specific[0] is None:
                st.session_state.data = run_sql(field_choice, schema_choice, table_choice, order_add(), limit_add())
                AgGrid(st.session_state.data)
                st.session_state.field_choice = field_choice     
                i += 1 

            # Performs the data pull when any of the buttons are clicked (run script and apply advanced)
            elif (run_script and filter_choice_specific_cleaned != []) or ((order_check or distinct_check) and apply_advanced):
                st.session_state.data = run_sql_with_where(field_choice, schema_choice, table_choice, filter_choice_specific_cleaned, order_add(), limit_add())
                AgGrid(st.session_state.data)
                st.session_state.field_choice = field_choice
                i += 1

            # Preserves the previously ran data pull should there be any changes in the fields
            elif not run_script and st.session_state.field_choice != []:
                AgGrid(st.session_state.data)
                i += 1

    with center_p1:

        if i == 0:
            st.title("Welcome! Now, let's get things running! 🏃‍♂️")
            st.write("Empty space, huh? Create selections in the sidebar then press the run button to generate your data. Your table will appear below.")

        elif i == 1:
            st.title("😮‍💨 Now that was a good run!")
            st.write("You could do another data pull, you know? Just change your selections in the sidebar.")

    with center_p3:

        rows_section = st.empty()
        col3, col4 = st.columns(2)

        if i == 1:

            with rows_section:
                rows_pulled = max(st.session_state.data["#"])
                st.caption(f"Number of rows pulled: **{rows_pulled}**")

            with col3:

                with st.expander("Curious how this works? Here's the code snippet. (It updates realtime!)"):
                    if field_choice == [] and filter_choice == []:
                        st.caption(f"No inputs yet.")

                    elif field_choice != [] and filter_choice == []:
                        st.caption(f"**Fields selected:** {field_list_cleaned}")
                        st.caption(f"**SQL Code:** SELECT {field_list_cleaned} FROM atlas.sale_insider {order_add()} {limit_add()}")

                    elif field_choice != [] and filter_choice != []:
                        st.caption(f"**Fields selected:** {field_list_cleaned}")
                        st.caption(f"**Filters selected:** {filter_choice_specific_cleaned}")
                        st.caption(f"**SQL Code:** SELECT {field_list_cleaned} FROM atlas.sale_insider WHERE {filter_choice_specific_cleaned} {order_add()} {limit_add()}")

            with col4:
                with st.expander("Ready to export your data?"):
                    data_download_name = st.text_input("Enter file name", placeholder="Press enter after specifiying the file name.", key="data_download")

                    @st.cache
                    def convert_format(data):
                        return data.to_csv(index=False)

                    data_download = convert_format(st.session_state.data)

                    st.download_button(
                        label = "Download as CSV file",
                        data = data_download,
                        file_name = f"{data_download_name}.csv"
                        )

        else:
            pass

st.sidebar.write(st.session_state.login_submit)