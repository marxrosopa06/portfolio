import math
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#PAGE CONFIGURATION
st.set_page_config(
     page_title="Marx's Portfolio",
     page_icon="📘",
     initial_sidebar_state="expanded",
     layout="wide")

with st.sidebar:
    algorithm = st.selectbox("Type of Algorithm:", ("(Intro)", "Linear Regression", "Logistic Regression", "Decision Tree / Random Forest"))

#INTRO
if algorithm == "(Intro)":
    st.header("⚡️ Welcome to my :orange[**Machine Learning Showcase**] page!")
    st.subheader("You may use the dropdown in the sidebar to select the type of algorithm that you would want to see.")
    st.subheader("Kindly just take note of the following things:")
    st.write("**1. Datasets that were used in the exercises were from :green[**Kaggle.com**]**")
    st.write("**2. Logistic Regression and Decision Tree/Random Forest have :green[**random_state set to 2**] for permanency of the data splits and accuracies since \n"
             ":green[**Hyperparameter Tuning using GridSearchCV and RandomizedSearchCV were utilized**]**")
    st.write("**3. Research is everything! :green[**I listed down the important references that I used as guides**] at the bottommost part of each of the pages**")


#LINEAR REGRESSION
elif algorithm == "Linear Regression":

    #INTRO
    st.header("📈 Machine Learning: Linear Regression")
    st.markdown("The goal of this exercise is to use :orange[**Linear Regression**] to predict someone's weight based on their height. \n"
                "We're using the :orange[**weight-height.csv dataset from Kaggle**]")
    st.caption("**Dataset**: https://www.kaggle.com/datasets/mustafaali96/weight-height")

    st.markdown("""---""")

    #PACKAGES AND DATASET
    with st.container():

        #Packages
        st.subheader("📍 Essentials")
        st.write("First things first, let's prepare and import the essential packages needed")
        st.code("import math \n"
                "import numpy as np \n"
                "import pandas as pd \n"
                "import seaborn as sns \n"
                "import streamlit as st \n"
                "import matplotlib.pyplot as plt \n"
                "from sklearn.linear_model import LinearRegression \n"
                "from sklearn import linear_model \n"
                "from sklearn.linear_model import Ridge \n"
                "from sklearn.linear_model import LogisticRegression \n"
                "from sklearn.model_selection import train_test_split \n"
                "from sklearn.metrics import mean_squared_error \n"
                "from sklearn.metrics import mean_absolute_error \n"
                "from sklearn.metrics import r2_score")

        #Dataset
        st.subheader("📍 Data Preparation")
        st.write("Let's now read the data and assign it to a variable")
        st.code("df = pd.read_csv('/Users/mariusrosopa/Desktop/Projects/Portfolio/datasets/weight-height.csv')")
        st.code("df.head(10)")

        df = pd.read_csv("/Users/mariusrosopa/Desktop/Projects/Portfolio/datasets/weight-height.csv")
        st.write(df.head(10))

    st.markdown("""---""")

    #EXPLORATORY DATA ANALYSIS
    with st.container():
        st.subheader("📍 Exploratory Data Analysis")
        st.write("We now explore the characteristics of our dataset")

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Data information")

        with col1:
            st.write(":orange[**Dimension**]: Data contains 10k observations with 3 columns")
            st.code("df.shape")
            st.write(df.shape)

        with col1:
            st.write(":orange[**Basic Statistics**]: General description of the data using various statistical metrics")
            st.code("df.describe()")
            st.write(df.describe())

        with col1:
            st.write(":orange[**Scatterplot of Height and Weight**]: We can see a linear trend, which further justifies that Linear Regression is the perfect \n"
            "algorithm for this dataset, but since it's still a bit scattered, it's good practice to check for :orange[**overfitting**], which we'll do later")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.scatterplot(data=df, x='height', y='weight', palette='mako')")
            fig = plt.figure(figsize=(10, 4))
            sns.scatterplot(data=df, x="height", y="weight", palette="mako")
            st.pyplot(fig)

        with col1:
            st.write(":orange[**Correlation**]: There's a high correlation between height and weight")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.heatmap(df.corr(), cmap='mako', annot=True) \n")
            fig = plt.figure(figsize=(10, 4))
            sns.heatmap(df.corr(), cmap="mako", annot=True)
            st.pyplot(fig)

        with col2:
            st.subheader("Handling disruptions")

        with col2:
            st.write(":orange[**Missing values**]: Data doesn't contain any missing values")
            st.code("df.isnull().sum().to_frame('NAs')")
            st.write(df.isnull().sum().to_frame('NAs'))

        with col2:
            st.markdown(":orange[**Outliers**]: Since the correlation between height and weight is already high and not much outliers are detected, \n"
            "skipping this part wouldn't hurt")

            st.caption(":green[**Height**]")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.boxplot(x=df['height'], palette='mako') \n")        
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=df["height"], palette="mako")
            st.pyplot(fig)      

            st.caption(":green[**Weight**]")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.boxplot(x=df['weight'], palette='mako') \n")        
            fig = plt.figure(figsize=(10, 4))
            sns.boxplot(x=df["weight"], palette="mako")
            st.pyplot(fig)      

    st.markdown("""---""")

    #MODELLING
    with st.container():
        st.subheader("📍 Modelling")
        st.subheader("1️⃣ Model Fitting")
        st.write("This is where we fit our training set into our model")
        st.write("Let's first split our dependent and independent variables into two separate DataFrames")
        
        st.code("height = pd.DataFrame(df['height']) \n"
                "weight = pd.DataFrame(df['weight']) \n")
        height = pd.DataFrame(df["height"])
        weight = pd.DataFrame(df["weight"])

        st.write("Then, we perform :orange[**Train-Test Split**] to split our data into training and testing sets for both dependent and independent variables")
        st.code("X_train, X_test, y_train, y_test = train_test_split(height, weight, test_size = 0.2)")
        X_train, X_test, y_train, y_test = train_test_split(height, weight, test_size = 0.2)

        st.write("Now that our data is ready, then we fit our training set to our model")
        st.code("model = LinearRegression().fit(X_train, y_train)")
        model = LinearRegression().fit(X_train, y_train)        

        st.markdown("""---""")
        
        #Overfit check
        st.subheader("✋ Check for overfitting")
        st.write("Before proceeding, let's first check for overfitting by using the :orange[**score()**] function for both training and test sets")
        st.code("score_train = model.score(X_train, y_train)")

        score_train = model.score(X_train, y_train)
        st.write("**Train set score (Linear):**", score_train)

        score_test = model.score(X_test, y_test)
        st.write("**Test set score (Linear):**", score_test)

        st.caption("➡ **Our model is :orange[**not exhibiting overfitting**], as the difference in accuracy of both train and test sets are very minimal, thus giving us the impression \n"
                   "that our model will perform well not just with our own data, but also with unforeseen data**.")

        #Lasso and Ridge
        st.subheader("😉 But wait up!")
        st.write("Since we didn't have an overfitted model, maybe it's not too bad to include two additional variants of Linear Regression, which are \n"
        ":orange[**Lasso and Ridge**]. Since both use penalizers, maybe we would achieve better accuracy compared to the usual Linear Regression.")

        col3, col4 = st.columns(2)

        with col3:
            st.caption(":green[**Lasso Regression**]")
            st.code("lasso_model = linear_model.Lasso(alpha=0.1) \n"
                    "lasso_model.fit(X_train, y_train) \n"
                    "lasso_score_train = lasso_model.score(X_train, y_train)")
            lasso_model = linear_model.Lasso(alpha=0.1)
            lasso_model.fit(X_train, y_train)

            st.code("lasso_score_train = lasso_model.score(X_train, y_train) \n"
                    "lasso_score_test = lasso_model.score(X_test, y_test)")
            lasso_score_train = lasso_model.score(X_train, y_train)
            lasso_score_test = lasso_model.score(X_test, y_test)
            st.write("**Train set score (Lasso)**:", lasso_score_train)
            st.write("**Test set score (Lasso)**:", lasso_score_test)

        with col4:
            st.caption(":green[**Ridge Regression**]")
            st.code("ridge_model = linear_model.Ridge(alpha=0.1) \n"
                    "ridge_model.fit(X_train, y_train) \n"
                    "ridge_score_train = ridge_model.score(X_train, y_train)")
            ridge_model = linear_model.Ridge(alpha=0.1)
            ridge_model.fit(X_train, y_train)

            st.code("ridge_score_train = ridge_model.score(X_train, y_train) \n"
                    "ridge_score_test = ridge_model.score(X_test, y_test")
            ridge_score_train = ridge_model.score(X_train, y_train)
            ridge_score_test = ridge_model.score(X_test, y_test)
            st.write("**Train set score (Ridge)**:", ridge_score_train)
            st.write("**Test set score (Ridge)**:", ridge_score_test)

        #Accuracy Scores Summary
        scores = [["Linear", score_train, score_test], ["Lasso", lasso_score_train, lasso_score_test], ["Ridge", ridge_score_train, ridge_score_test]]
        scores_df = pd.DataFrame(scores, columns=["Model", "Train", "Test"])
        st.write(scores_df)
        st.caption("➡ **Tabulating all the scores, it seems that there's really not much difference if we use Lasso or Ridge compared to the usual Linear Regression. \n"
                   "It was still worth the try! 😅. From here, we'll just proceed with :orange[**Linear Regression**]**.")

        st.markdown("""---""")

        #Model Evaluation
        st.subheader("2️⃣ Model Evaluation")
        st.write("Let's now predict and evaluate our model using a :orange[**various evaluation metrics**]")
        st.code("weight_pred = model.predict(height) \n"
                "df_weight_pred = pd.DataFrame(weight_pred, columns=['Predicted Weight'] \n"
                "df_new = pd.concat([df, df_weight_pred], axis=1, join='inner') \n"
                "st.write(df_new.head(10))")

        weight_pred = model.predict(height)
        df_weight_pred = pd.DataFrame(weight_pred, columns=["weight_pred"])
        df_new = pd.concat([df, df_weight_pred], axis=1, join="inner")
        st.write(df_new.head(100))
        
        st.subheader("⚡️ True Weight vs. Predicted Weight")
        st.write(":blue[**Blue**] is the true weight while :orange[**Orange**] is the predicted weight")
        fig = plt.figure(figsize=(19, 5))
        sns.lineplot(data=df_new.drop("height", axis=1))
        # ax = sns.relplot(kind="line", x="height", y="weight", data=df_new, height=4, aspect=3.5)
        # ax.map_dataframe(sns.lineplot, "height", "weight_pred", color='r')
        st.pyplot(fig)

        st.write("Let's now check the accuracy of our model through various evaluation metrics such as :orange[**R2 Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), \n"
                 "Mean Absolute Error (MAE)**]")

        st.code("r2_score(df_new['weight'], df_new['weight_pred']) \n"
                "mean_squared_error(df_new['weight'], df_new['weight_pred']) \n"
                "math.sqrt(mean_squared_error(df_new['weight'], df_new['weight_pred'])) \n"
                "mean_absolute_error(df_new['weight'], df_new['weight_pred'])")

        st.write("**R2 score**:", r2_score(df_new["weight"], df_new["weight_pred"]))
        st.write("**MSE**:", mean_squared_error(df_new["weight"], df_new["weight_pred"]))
        st.write("**RMSE**:", math.sqrt(mean_squared_error(df_new["weight"], df_new["weight_pred"])))
        st.write("**MAE**:", mean_absolute_error(df_new["weight"], df_new["weight_pred"]))

        st.caption("➡ **:orange[**We got a high R2 score for our model, which is a good sign of a good predictive model**]. For the other error metrics, we won't be able to interpret \n"
                   "them without comparing them to the same metrics of a different model**.")
        
        st.markdown("""---""")

        #Model Comparison
        st.subheader("3️⃣ Model Comparison")
        st.write("Say if we remove the outliers that we detected a while ago, would that perform better than our produced model? Let's find out")

        st.write("Let's produce our boxplots again for :orange[**height and weight**]")

        col5, col6 = st.columns(2)

        with col5:
            st.caption(":green[**Height**]")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.boxplot(x=df['height'], palette='mako')")
            fig = plt.figure(figsize=(10, 4)) 
            sns.boxplot(x=df['height'], palette='mako')
            st.pyplot(fig)

        with col6:
            st.caption(":green[**Weight**]")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.boxplot(x=df['weight'], palette='mako')")
            fig = plt.figure(figsize=(10, 4)) 
            sns.boxplot(x=df['weight'], palette='mako')
            st.pyplot(fig)

        st.write("Let's create a function that would extract the outliers specified in the boxplots")
        st.code("def detect_outlier(data): \n"
                "   #Get the Q1 and Q3 values \n"
                "   q1, q3 = np.percentile(sorted(data), [25, 75]) \n"
                " \n"
                "   #Interquartile Range \n"
                "   iqr = q3-q1 \n"
                " \n"
                "   #Get the lower and upper bounds \n"
                "   lower_bound = q1 - (1.5*iqr) \n"
                "   upper_bound = q3 + (1.5*iqr) \n"
                " \n"
                "   #Get the outliers \n"
                "   outliers = [x for x in data if x <= lower_bound or x >= upper_bound] \n"
                " \n"
                "   return outliers")

        def detect_outlier(data):
            #Get the Q1 and Q3 values
            q1, q3 = np.percentile(sorted(data), [25, 75])

            #Interquartile Range
            iqr = q3-q1

            #Get the lower and upper bounds
            lower_bound = q1 - (1.5*iqr)
            upper_bound = q3 + (1.5*iqr)

            #Get the outliers
            outliers = [x for x in data if x <= lower_bound or x >= upper_bound]

            return outliers

        col7, col8 = st.columns(2)

        with col7:
            st.caption(":green[**Height**]")
            st.code("#Use the function for Height to get the outliers \n"
                    "detect_outlier(df['height']) \n"
                    " \n"
                    "#Put the outliers in a separate variable \n"
                    "index_height = df[df['height'].isin(detect_outlier(df['height']))].index \n")
            st.write(detect_outlier(df["height"]))

            index_height = df[df["height"].isin(detect_outlier(df["height"]))].index

        with col8:
            st.caption(":green[**Weight**]")
            st.code("#Use the function for Weight to get the outliers \n"
                    "detect_outlier(df['weight']) \n"
                    " \n"
                    "#Put the outliers in a separate variable \n"
                    "index_weight = df[df['weight'].isin(detect_outlier(df['weight']))].index \n")
            
            st.write(detect_outlier(df["weight"]))

            index_weight = df[df["weight"].isin(detect_outlier(df["weight"]))].index

        st.write("Since the outlier in Weight is also within the same row of one of the outliers in Height, we'll just drop the rows with outliers in Height")

        df_no_out = df.drop(index_height)

        st.code("#Drop the outliers using their indexes \n"
                "outliers_height_df = df.drop(height)")
        st.write("**New shape without the outliers**: ", df_no_out.shape)

        st.write("Let's now do the modelling without the outliers")
        st.caption("**Model Fitting**")
        st.code("model_no_out = LinearRegression().fit(X_train, y_train) \n"
                "weight_pred_no_out = model.predict(height) \n"
                "df_weight_pred_no_out = pd.DataFrame(weight_pred_no_out, columns=['weight_pred']) \n"
                "df_new = pd.concat([df_no_out, df_weight_pred_no_out], axis=1, join='inner') \n")

        model_no_out = LinearRegression().fit(X_train, y_train)
        weight_pred_no_out = model.predict(height)
        df_weight_pred_no_out = pd.DataFrame(weight_pred_no_out, columns=["weight_pred"])
        df_new_no_out = pd.concat([df_no_out, df_weight_pred_no_out], axis=1, join="inner")
        st.write(df_new.head(100))

        st.caption("**Model Evaluation**")

        col9, col10 = st.columns(2)

        with col9:
            st.code("r2_score(df_new_no_out['weight'], df_new_no_out['weight_pred']) \n"
                    "mean_squared_error(df_new_no_out['weight'], df_new_no_out['weight_pred']) \n"
                    "math.sqrt(mean_squared_error(df_new_no_out['weight'], df_new_no_out['weight_pred'])) \n"
                    "mean_absolute_error(df_new_no_out['weight'], df_new_no_out['weight_pred'])")
            
            st.caption(":green[**Evaluation metrics WITHOUT the outliers**]")
            st.write("**R2 score**:", r2_score(df_new_no_out["weight"], df_new_no_out["weight_pred"]))
            st.write("**MSE**:", mean_squared_error(df_new_no_out["weight"], df_new_no_out["weight_pred"]))
            st.write("**RMSE**:", math.sqrt(mean_squared_error(df_new_no_out["weight"], df_new_no_out["weight_pred"])))
            st.write("**MAE**:", mean_absolute_error(df_new_no_out["weight"], df_new_no_out["weight_pred"]))

        with col10:
            st.code("r2_score(df_new['weight'], df_new['weight_pred']) \n"
                    "mean_squared_error(df_new['weight'], df_new['weight_pred']) \n"
                    "math.sqrt(mean_squared_error(df_new['weight'], df_new['weight_pred'])) \n"
                    "mean_absolute_error(df_new['weight'], df_new['weight_pred'])")

            st.caption(":violet[**Evaluation metrics WITH the outliers (Existing Model)**]")
            st.write("**R2 score**:", r2_score(df_new["weight"], df_new["weight_pred"]))
            st.write("**MSE**:", mean_squared_error(df_new["weight"], df_new["weight_pred"]))
            st.write("**RMSE**:", math.sqrt(mean_squared_error(df_new["weight"], df_new["weight_pred"])))
            st.write("**MAE**:", mean_absolute_error(df_new["weight"], df_new["weight_pred"]))

        st.caption("➡ **The :orange[**existing model seems to be performing slightly better than the model trained with a dataset without the outliers**] because of \n"
                   "the higher R2 score. This is also evident in terms of the error metrics because the existing model has slightly lower values**")


elif algorithm == "Logistic Regression":

    #INTRO
    st.header("👯‍♀️ Machine Learning: Logistic Regression")
    st.markdown("The goal of this exercise is to use :orange[**Logistic Regression**] to predict someone's capability of clicking an ad with the use of different user features. \n"
                "We're using the :orange[**advertising.csv dataset from Kaggle**]")
    st.caption("**Dataset**: https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad")

    st.markdown("""---""")

    #PACKAGES AND DATASET
    with st.container():

        #Packages
        st.subheader("📍 Essentials")
        st.write("First things first, let's prepare and import the essential packages needed")
        st.code("import math \n"
                "import numpy as np \n"
                "import pandas as pd \n"
                "import seaborn as sns \n"
                "import streamlit as st \n"
                "import matplotlib.pyplot as plt \n"
                "import warnings \n"
                "from sklearn.linear_model import LogisticRegression \n"
                "from sklearn.model_selection import train_test_split \n"
                "from sklearn.model_selection import GridSearchCV \n")

        #Dataset
        st.subheader("📍 Data Preparation")
        st.write("Let's now read the dataset and assign it to a variable")
        st.code("df = pd.read_csv('/Users/mariusrosopa/Desktop/Projects/Portfolio/datasets/insurance_data.csv')")
        st.code("df.head(10)")

        df = pd.read_csv("/Users/mariusrosopa/Desktop/Projects/Portfolio/datasets/advertising.csv")
        df.index = np.arange(1, len(df) + 1)
        st.write(df.head(10))

    st.markdown("""---""")

    #EXPLORATORY DATA ANALYSIS
    with st.container():
        st.subheader("📍 Exploratory Data Analysis")
        st.write("We now explore the characteristics of our dataset")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data information")

        with col1:
            st.write(":orange[**Dimension**]: Data contains 1000 observations with 10 columns")
            st.code("df.shape")
            st.write(df.shape)

        with col1:
            st.write(":orange[**Basic Statistics**]: General description of the data using various statistical metrics")
            st.code("df.describe()")
            st.write(df.describe())

        with col1:
            st.write(":orange[**Correlation**]: There seems to be no significant correlation among the features available; this means that there's no possibility of encountering multicollinearity")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.heatmap(df.corr(), cmap='mako', annot=True)")
            fig = plt.figure(figsize=(10, 4))
            sns.heatmap(df.corr(), cmap="mako", annot=True)
            st.pyplot(fig)

        with col2:
            st.subheader("Feature selection")

        with col2:
            st.write("Let's review the features of the data first. before proceeding to the model fitting. Upon observation, :orange[**we can see that there are some features \n"
                     "that could be dropped for simplification**]. I'm choosing to drop :green[**ad_topic_line, city, country, and timestamp**] because they don't describe"
                     "the users as much as the other remaining features")
            st.code("df.head(5)")
            st.write(df.head(5))

        with col2:
            st.write("Let's set a new variable only containing the relevant features")
            st.code("new_df = df.drop(['ad_topic_line', 'city', 'country', 'timestamp'], axis=1) \n"
                    "new_df.head(5)")
            new_df = df.drop(['ad_topic_line', 'city', 'country', 'timestamp'], axis=1)
            st.write(new_df.head(5))

        st.markdown("""---""")

        #Handling Disruptions
        st.subheader("Handling disruptions")

        st.write(":orange[**Missing values**]: Data doesn't contain any missing values")
        st.code("new_df.isnull().sum().to_frame('NAs')")
        st.write(new_df.isnull().sum().to_frame('NAs'))

        st.write(":orange[**Outliers**]: Out of all the features, only **area_income** has outliers; let's further investigate it.")

        col3, col4 = st.columns(2)
        
        with col3:
            st.caption(":green[**daily_time_spent_on_site**]")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.boxplot(x=df['daily_time_spent_on_site'], palette='mako')")        
            fig = plt.figure(figsize=(10, 2))
            sns.boxplot(x=df["daily_time_spent_on_site"], palette="mako")
            st.pyplot(fig)

            st.caption(":green[**age**]")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.boxplot(x=df['age'], palette='mako')")        
            fig = plt.figure(figsize=(10, 2))
            sns.boxplot(x=df["age"], palette="mako")
            st.pyplot(fig)      

        with col4:
            st.caption(":green[**area_income**]")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.boxplot(x=df['area_income'], palette='mako')")        
            fig = plt.figure(figsize=(10, 2))
            sns.boxplot(x=df["area_income"], palette="mako")
            st.pyplot(fig)

            st.caption(":green[**daily_internet_usage**]")
            st.code("fig = plt.figure(figsize=(10, 4)) \n"
                    "sns.boxplot(x=df['daily_internet_usage'], palette='mako')")        
            fig = plt.figure(figsize=(10, 2))
            sns.boxplot(x=df["daily_internet_usage"], palette="mako")
            st.pyplot(fig)      

        st.write(":orange[**Outlier handling for area_income**]: Let's show the data that contains outliers under this feature")
        st.code("new_df[new_df['area_income']<20000]")
        st.write(new_df[new_df["area_income"]<20000])
        st.caption("➡ **Based on my observation, the numbers seem to be valid inputs and not errors, so keeping them would be the logical choice, but it would be best to \n"
                 ":orange[**create two seperate models: One where it contains all data points and one where these outliers are removed**], to see if the accuracy improves upon removal \n"
                 "of these data points**.")
        
        st.write("For this scenario, let's create a separate variable where it contains the data without the presence of the outliers.")
        st.code("new_df_no_outliers = new_df[new_df['area_income'>20000]]")
        new_df_no_outliers = new_df[new_df["area_income"]>=20000]
        st.write(new_df_no_outliers)

    st.markdown("""---""")

    #MODELLING
    st.subheader("📍 Modelling")
    st.subheader("1️⃣ Model Fitting (with outliers vs. w/o outliers)")
    st.write("This is where we fit our training sets into our models and adjust the parameters for better accuracy")
    st.write("Let's first split our dependent and independent variables into two separate DataFrames")

    col5, col6 = st.columns(2)

    with col5:
        st.caption(":green[**With Outliers**]")

    with col6:
        st.caption(":green[**Without Outliers**]")

    col7, col8 = st.columns(2)

    with col7:
        st.code("X = pd.DataFrame(new_df.drop('clicked_on_ad', axis=1)) \n"
                "y = pd.DataFrame(new_df['clicked_on_ad'])")
        X = pd.DataFrame(new_df.drop("clicked_on_ad", axis=1))
        y = pd.DataFrame(new_df["clicked_on_ad"])

    with col8:
        st.code("X_no_out = pd.DataFrame(new_df_no_outliers.drop('clicked_on_ad', axis=1)) \n"
                "y_no_out = pd.DataFrame(new_df_no_outliers['clicked_on_ad'])")
        X_no_out = pd.DataFrame(new_df_no_outliers.drop("clicked_on_ad", axis=1))
        y_no_out = pd.DataFrame(new_df_no_outliers["clicked_on_ad"])
        
    st.write("Then we perform :orange[**Train-Test Split**] to split our data into training and testing sets for both dependent and independent variables")

    col9, col10 = st.columns(2)

    with col9:
        st.caption(":green[**With Outliers**]")

    with col10:
        st.caption(":green[**Without Outliers**]")

    col11, col12 = st.columns(2)

    with col11:
        st.code("with warnings.catch_warnings(): \n"
                "   warnings.simplefilter('ignore') \n"
                " \n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=2)")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=2)

    with col12:
        st.code("with warnings.catch_warnings(): \n"
                "   warnings.simplefilter('ignore') \n"
                " \n"
                "X_train_no_out, X_test_no_out, y_train_no_out, y_test_no_out = train_test_split(X_no_out, y_no_out, test_size = 0.1, random_state=2)")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            X_train_no_out, X_test_no_out, y_train_no_out, y_test_no_out = train_test_split(X_no_out, y_no_out, test_size = 0.1, random_state=2)

    st.write("Now that our data is ready, we fit our training sets to our model")

    col13, col14 = st.columns(2)

    with col13:
        st.caption(":green[**With Outliers**]")

    with col14:
        st.caption(":green[**Without Outliers**]")

    col15, col16 = st.columns(2)

    with col15:
        st.code("model = LogisticRegression().fit(X_train, y_train)")
        model = LogisticRegression().fit(X_train, y_train)

    with col16:
        st.code("model_no_out = LogisticRegression().fit(X_train_no_out, y_train_no_out)")
        model_no_out = LogisticRegression().fit(X_train_no_out, y_train_no_out)

    st.markdown("""---""")

    #Overfit check
    st.subheader("✋ Check for overfitting")
    st.write("Before proceeding, let's first check for overfitting by using the :orange[**score()**] function for both training and test sets")

    col17, col18 = st.columns(2)

    with col17:
        st.caption(":green[**With Outliers**]")

    with col18:
        st.caption(":green[**Without Outliers**]")

    col19, col20 = st.columns(2)

    with col19:
        st.code("score_train = model.score(X_train, y_train) \n"
                "score_test = model.score(X_test, y_test)")

        score_train = model.score(X_train, y_train)
        st.write("**Train set score:**", score_train)
        score_test = model.score(X_test, y_test)
        st.write("**Test set score:**", score_test)

    with col20:
        st.code("score_train_no_out = model.score(X_train_no_out, y_train_no_out) \n"
                "score_test_no_out = model.score(X_test_no_out, y_test_no_out)")

        score_train_no_out = model.score(X_train_no_out, y_train_no_out)
        st.write("**Train set score:**", score_train_no_out)
        score_test_no_out = model.score(X_test_no_out, y_test_no_out)
        st.write("**Test set score:**", score_test_no_out)

    st.caption("➡ **The differences in the scores are not that drastic, so let's assume that there's :orange[**no sign of overfitting on these datasets**], but it would be best \n"
               "to perform :orange[**Hyperparameter Tuning**] to really bring out the best parameters that would produce the models with the highest accuracy and \n"
               "to finally compare which of the two (with vs. without outliers) has the better accuracy**.")

    st.markdown("""---""")

    #GridSearchCV
    st.subheader("⚡️ Hyperparameter Tuning with GridSearchCV")
    st.write("This is where we introduce the function :orange[**GridSearchCV**] to iterate and test different combinations of parameters")

    param_grid = {
        "penalty":  ["l1", "l2"],
        "C":        np.logspace(-3,3,7),
        "solver":   ["lbfgs", "liblinear", "newton-cg"]
    }

    st.write("Set up the parameter grid")
    st.code("param_grid = { \n"
        "     'penalty':  ['l1', 'l2'], \n"
        "     'C':        np.logspace(-3,3,7), \n"
        "     'solver':   ['lbfgs', 'liblinear', 'newton-cg'] \n"
    "}")
    st.caption("**Penalty:** Regularization techniques that we could apply to our model. This solves overfitting by penalizing the loss function. L1 adds the absolute value of magnitude, while L2 adds the squared magnitude. \n"
    "| **C:** Strength of the regularization; lower means stronger regularization. | **Solver:** Type of algorithm to be performed; solvers have different characteristics such as better for smaller datasets, faster processing for larger datasets, etc.")

    col21, col22 = st.columns(2)

    with col21:
        st.caption(":green[**With Outliers**]")

    with col22:
        st.caption(":green[**Without Outliers**]")

    st.write("This is where we fit our model again and feed it into :orange[**GridSearchCV**], while using the parameters we indicated above")

    col23, col24 = st.columns(2)

    with col23:
        st.code("with warnings.catch_warnings(): \n"
                "   warnings.simplefilter('ignore') \n"
                " \n"
                "   logreg = LogisticRegression() \n"
                "   clf = GridSearchCV(logreg, param_grid=param_grid, scoring='accuracy', cv=10) \n"
                "   clf.fit(X_train, y_train)")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            logreg = LogisticRegression()
            clf = GridSearchCV(logreg, param_grid=param_grid, scoring="accuracy", cv=10)
            clf.fit(X_train, y_train)

    with col24:
        st.code("with warnings.catch_warnings(): \n"
                "   warnings.simplefilter('ignore') \n"
                " \n"
                "   logreg_no_out = LogisticRegression() \n"
                "   clf_no_out = GridSearchCV(logreg_no_out, param_grid=param_grid, scoring='accuracy', cv=10) \n"
                "   clf_no_out.fit(X_train_no_out, y_train_no_out)")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            logreg_no_out = LogisticRegression()
            clf_no_out = GridSearchCV(logreg_no_out, param_grid=param_grid, scoring="accuracy", cv=10)
            clf_no_out.fit(X_train_no_out, y_train_no_out)

    st.write("Let's now check the best parameter combination and its accuracy")
    
    col25, col26 = st.columns(2)

    with col25:
        st.caption(":green[**With Outliers**]")
        st.code("clf.best_params_ \n"
                "clf.best_score_")
        st.write("**Tuned Hyperparameters:**", clf.best_params_)
        st.write(":orange[**Accuracy:**]", clf.best_score_)

    with col26:
        st.caption(":green[**Without Outliers**]")
        st.code("clf_no_out.best_params_ \n"
                "clf_no_out.best_score_")
        st.write("**Tuned Hyperparameters:**", clf_no_out.best_params_)
        st.write(":orange[**Accuracy:**]", clf_no_out.best_score_)

    st.caption("➡ **We've interpreted earlier that there's no overfitting when accuracies of the train and test sets were compared, but it seems that there was! :orange[**There's \n"
                "a drastic improvement in the accuracies when we reduced the C (hyperparameter) to 0.001 (which is defaulted at 1)**]; a smaller C attributes to a stronger \n"
                "regularization strength. Also, another notable thing is there's not much difference in accuracy between the with and without outliers dataset. \n"
                ":orange[**With the dataset containing the outliers having a slightly higher accuracy, we'll proceed with only that instead**]**.")
    
    accuracies_data = {"With Outliers": [score_train, clf.best_score_], "Without Outliers": [score_train_no_out, clf_no_out.best_score_]}
    accuracies = pd.DataFrame(accuracies_data, index=["Without GridSearchCV", "With GridSearchCV"])
    st.write(accuracies)
    
    st.write("Let's now use the best parameters generated by the :orange[**GridSearchCV**] for our dataset with the outliers and do our model fitting.")
    st.code("model = LogisticRegression(penalty='l2', C=0.001, solver='newton-cg').fit(X_train, y_train)")
    model = LogisticRegression(penalty="l2", C=0.001, solver="newton-cg").fit(X_train, y_train)

    st.markdown("""---""")

    #Model Prediction
    st.subheader("2️⃣ Model Evaluation")
    st.write("Let's now predict and evaluate our model using a :orange[**Confusion Matrix and a Classification Report**]")

    col27, col28 = st.columns(2)

    with col27:
        st.caption(":green[**True vs. Predicted values for clicked_on_ad**]")
        st.code("y_pred = model.predict(X_test) \n"
                "df_clicked_on_ad_pred = pd.DataFrame(y_pred, columns=['clicked_on_ad_pred']) \n"
                "df_preds = pd.concat([y_test.reset_index().drop('index', axis=1), df_clicked_on_ad_pred], axis=1, join='inner') \n"
                "df_preds.rename(columns={'clicked_on_ad': 'clicked_on_ad_true'}) \n")
        
        y_pred = model.predict(X_test)
        df_clicked_on_ad_pred = pd.DataFrame(y_pred, columns=["clicked_on_ad_pred"])
        df_preds = pd.concat([y_test.reset_index().drop("index", axis=1), df_clicked_on_ad_pred], axis=1, join="inner")
        df_preds.rename(columns={"clicked_on_ad": "clicked_on_ad_true"})
        st.write(df_preds.rename(columns={"clicked_on_ad": "clicked_on_ad_true"}).head(10))

    with col28:
        st.caption(":green[**Confusion Matrix for True vs. Predicted values**]")
        st.code("cm = confusion_matrix(y_test, y_pred) \n"
                "fig = plt.figure(figsize=(8, 6)) \n"
                "ax = plt.subplot() \n"
                "sns.heatmap(cm, cmap='mako', annot=True, ax=ax) \n"
                " \n"
                "ax.set_xlabel('True labels') \n"
                "ax.set_ylabel('Predicted labels') \n"
                "ax.xaxis.set_ticklabels(['clicked_on_ad = 0', 'clicked_on_ad = 1']) \n"
                "ax.yaxis.set_ticklabels(['clicked_on_ad = 0', 'clicked_on_ad = 1'])")
        
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        sns.heatmap(cm, cmap="mako", annot=True, ax=ax)

        ax.set_xlabel('True labels')
        ax.set_ylabel('Predicted labels')
        ax.xaxis.set_ticklabels(['clicked_on_ad = 0', 'clicked_on_ad = 1'])
        ax.yaxis.set_ticklabels(['clicked_on_ad = 0', 'clicked_on_ad = 1'])

        st.pyplot(fig)

    st.caption("➡ **In evaluating a Logistic Regression, it's important to produce a Confusion Matrix, which we did in the previous \n"
               "section. With that, :orange[**calculating for the Accuracy, Precision, Recall, and F1 score are very important**] so we could be able to \n"
               "interpret the Confusion Matrix; this can be done using the :orange[**classification_report()**] function**.")
    
    col29, col30 = st.columns(2)

    with col29:
        st.caption(":green[**Classification Report**]")
        st.code("report = classification_report(y_test, y_pred, output_dict=True) \n"
                "df_report = pd.DataFrame(report).transpose()")
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.write(df_report)
        st.caption("➡ **All of the metrics in our generated Classification Report show high values, almost approaching the perfect score of 1, which :orange[**indicates \n"
                    "a highly accurate classification model**]**.")

    with col30:
        st.caption(":green[**A brief summary of the metrics in the Classification Report**]")
        st.write(":orange[**Precision**] is the ability of a classifier not to label an instance positive that is actually negative. \n"
                    "For each class, it is defined as the ratio of true positives to the sum of a true positive and false positive. It asks the question: \n"
                    "What percent of your predictions were correct?")
        st.write(":orange[**Recall**] is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of \n"
                    "true positives to the sum of true positives and false negatives. It asks the question: What percent of the positive cases did you catch?")
        st.write(":orange[**F1 score**] is the weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. \n"
                    "F1 scores are lower than accuracy measures as they embed precision and recall into their computation. It asks the question: \n"
                    "What percent of positive predictions were correct?")
        st.write(":orange[**Support**] is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training \n"
                    "data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling \n"
                    "or rebalancing.")
        st.caption("**Definitions were gathered from** https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397")

    st.markdown("""---""")

    #Model Comparison
    st.subheader("3️⃣ Model Comparison")
    st.write("Remember that we abandoned the dataset without the outliers? To get a clearer understanding of the performance of our produced model, \n"
                "it's always best to have another model to compare it to; hence, :orange[**we're going to revive it and compare its performance with our existing one**].")
    
    st.caption("**Model Fitting**")
    st.code("model_no_out = LogisticRegression(penalty='l2', C=0.001, solver='newton-cg').fit(X_train_no_out, y_train_no_out)")
    model_no_out = LogisticRegression(penalty="l2", C=0.001, solver="newton-cg").fit(X_train_no_out, y_train_no_out)

    st.caption("**Model Prediction**")

    col31, col32 = st.columns(2)

    with col31:
        st.caption(":green[**True vs. Predicted values for clicked_on_ad for the dataset without the outliers**]")
        st.code("y_pred = model.predict(X_test) \n"
                "df_clicked_on_ad_pred = pd.DataFrame(y_pred, columns=['clicked_on_ad_pred']) \n"
                "df_preds = pd.concat([y_test.reset_index().drop('index', axis=1), df_clicked_on_ad_pred], axis=1, join='inner') \n"
                "df_preds.rename(columns={'clicked_on_ad': 'clicked_on_ad_true'}) \n")
        
        y_pred_no_out = model.predict(X_test_no_out)
        df_clicked_on_ad_pred_no_out = pd.DataFrame(y_pred_no_out, columns=["clicked_on_ad_pred"])
        df_preds_no_out = pd.concat([y_test_no_out.reset_index().drop("index", axis=1), df_clicked_on_ad_pred_no_out], axis=1, join="inner")
        df_preds_no_out.rename(columns={"clicked_on_ad": "clicked_on_ad_true"})
        st.write(df_preds_no_out.rename(columns={"clicked_on_ad": "clicked_on_ad_true"}).head(10))

    with col32:
        st.caption(":green[**Confusion Matrix for True vs. Predicted values**]")
        st.code("cm = confusion_matrix(y_test, y_pred) \n"
                "fig = plt.figure(figsize=(8, 6)) \n"
                "ax = plt.subplot() \n"
                "sns.heatmap(cm, cmap='mako', annot=True, ax=ax) \n"
                " \n"
                "ax.set_xlabel('True labels') \n"
                "ax.set_ylabel('Predicted labels') \n"
                "ax.xaxis.set_ticklabels(['clicked_on_ad = 0', 'clicked_on_ad = 1']) \n"
                "ax.yaxis.set_ticklabels(['clicked_on_ad = 0', 'clicked_on_ad = 1'])")
        
        cm_no_out = confusion_matrix(y_test_no_out, y_pred_no_out)
        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        sns.heatmap(cm_no_out, cmap="mako", annot=True, ax=ax)

        ax.set_xlabel('True labels')
        ax.set_ylabel('Predicted labels')
        ax.xaxis.set_ticklabels(['clicked_on_ad = 0', 'clicked_on_ad = 1'])
        ax.yaxis.set_ticklabels(['clicked_on_ad = 0', 'clicked_on_ad = 1'])

        st.pyplot(fig)

    st.caption("**Model Evaluation**")

    col33, col34 = st.columns(2)

    with col33:
        st.caption(":green[**Classification Report WITHOUT the outliers**]")
        st.code("report_no_out = classification_report(y_test_no_out, y_pred_no_out, output_dict=True) \n"
                "df_report_no_out = pd.DataFrame(report_no_out).transpose()")
        report_no_out = classification_report(y_test_no_out, y_pred_no_out, output_dict=True)
        df_report_no_out = pd.DataFrame(report_no_out).transpose()
        st.write(df_report_no_out)

    with col34:
        st.caption(":violet[**Classification Report WITH the outliers (Existing Model)**]")
        st.code("report = classification_report(y_test, y_pred, output_dict=True) \n"
                "df_report = pd.DataFrame(report).transpose()")
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.write(df_report)

    st.caption("➡ **Simply looking at the metrics, we can see that :orange[**our existing model is the better model**]**.")

    st.markdown("""---""")

    st.caption("**Important references:**")

    st.caption("https://towardsdatascience.com/dont-sweat-the-solver-stuff-aea7cddc3451")
    st.caption("https://pub.towardsai.net/binary-classification-using-logistic-regression-vs-visualizations-8b9b5dce8e8b")
    st.caption("https://www.analyticsvidhya.com/blog/2021/09/guide-for-building-an-end-to-end-logistic-regression-model/")
    st.caption("https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397")

elif algorithm == "Decision Tree / Random Forest":

    #INTRO
    st.header("🌳 Machine Learning: Decision Tree & Random Forest")
    st.markdown("The goal of this exercise is to use :orange[**Decision Tree and Random Forest**] to predict someone's capability of having their loan approved given a set of the person's attributes. \n"
                "We're going to use the :orange[**loan prediction dataset from Kaggle**]")
    st.caption("**Dataset**: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?select=train_u6lujuX_CVtuZ9i.csv")

    st.markdown("""---""")

    #PACKAGES AND DATASET
    with st.container():

        #Packages
        st.subheader("📍 Essentials")
        st.write("First things first, let's prepare and import the essential packages needed")
        st.code("import math \n"
                "import numpy as np \n"
                "import pandas as pd \n"
                "import seaborn as sns \n"
                "import streamlit as st \n"
                "import matplotlib.pyplot as plt \n"
                "from sklearn.linear_model import LogisticRegression \n"
                "from sklearn.model_selection import train_test_split \n"
                "import statsmodels.api as sm \n"
                "from sklearn.metrics import mean_squared_error \n"
                "from sklearn.metrics import mean_absolute_error \n")

        #Dataset
        st.subheader("📍 Data Preparation")
        st.write("Let's now read the dataset and assign it to variables")
        st.code("df = pd.read_csv('/Users/mariusrosopa/Desktop/Projects/Portfolio/datasets/train_u6lujuX_CVtuZ9i.csv') \n"
                "df.index = np.arange(1, len(df_train) + 1)")
        
        df = pd.read_csv("/Users/mariusrosopa/Desktop/Projects/Portfolio/datasets/train_u6lujuX_CVtuZ9i.csv")
        df.index = np.arange(1, len(df) + 1)

        st.code("df.head(10)")
        st.write(df.head(10))

        st.write("Before proceeding, let's first drop the :orange[**Loan_ID**] column because we're not going to use it as a feature since it's a unique column")
        st.code("df.drop(['Loan_ID'], axis=1, inplace=True) \n"
                "df.head(5)")
        df.drop(["Loan_ID"], axis=1, inplace=True)
        st.write(df.head(5))

    st.markdown("""---""")

    #EXPLORATORY DATA ANALYSIS
    with st.container():
        st.subheader("📍 Exploratory Data Analysis")
        st.write("We now explore the characteristics of our dataset")

        st.write(":orange[**Dimension**]: The dataset contains 614 observations with 13 features")
        st.code("df.shape")
        st.write(df.shape)

        st.write(":orange[**Basic Statistics**]: General description of the numerical features using various statistical metrics")
        st.code("df.describe()")
        st.write(df.describe())

        st.write(":orange[**Value Counts**]: Observing the raw data, I noticed that there are some features that are categorical in nature. It would be best to convert them to \n"
                 "indicator variables so they could be used in building the model with the use of the :orange[**get_dummies function from Pandas**], which we'll do \n"
                 "later on. The features that are categorical in nature are :green[**Gender, Married, Dependents, Education, Self_Employed, Loan_Amount_Term, Credit_History, \n"
                 "Property_Area, and Loan_Status**]")

        col1, col2 = st.columns(2)

        with col1:
            st.caption(":green[**Gender**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Gender', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Gender", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

            st.caption(":green[**Married**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Married', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Married", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

            st.caption(":green[**Dependents**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Dependents', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Dependents", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

            st.caption(":green[**Education**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Education', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Education", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

            st.caption(":green[**Self_Employed**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Self_Employed', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Self_Employed", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

        with col2:
            st.caption(":green[**Loan_Amount_Term**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Loan_Amount_Term', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Loan_Amount_Term", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

            st.caption(":green[**Credit_History**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Credit_History', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Credit_History", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

            st.caption(":green[**Property_Area**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Property_Area', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Property_Area", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

            st.caption(":green[**Loan_Status**]")
            st.code("fig = plt.figure(figsize=(9,6)) \n"
                    "ax = sns.countplot(data=df, x='Loan_Status', palette='mako') \n"
                    "ax.bar_label(ax.containers[0]) \n"
                    "st.pyplot(fig)")
            fig = plt.figure(figsize=(9,6))
            ax = sns.countplot(data=df, x="Loan_Status", palette="mako")
            ax.bar_label(ax.containers[0])
            st.pyplot(fig)

        st.markdown("""---""")

        #Handling Disruptions
        st.subheader("📍 Handling Disruptions")
        st.write(":orange[**Missing values**]: Both datasets have missing values in different features. In handling missing values, the easiest way to do it is by imputation \n"
                 "according to the nature of the feature. :green[**Missing categorical observations are often handled by imputing using the Mode, while \n"
                 "missing numerical variables are often replaced by the Mean**]")

        st.code("df_missing = df.isnull().sum().to_frame('NAs') \n"
                "df_missing = df_missing[df_missing['NAs']!=0].reset_index().rename(columns={'index':'feature'})")
        df_missing = df.isnull().sum().to_frame('NAs')
        df_missing = df_missing[df_missing["NAs"]!=0].reset_index().rename(columns={"index":"feature"})
        st.write(df_missing)
            
        st.caption("**Let's tabulate the features and get the necessary values that we'll be using for the missing values imputation**")

        st.code("df_missing_var = [] \n"
                "df_missing_imp = [] \n"
                "df_missing_type = [] \n"
                " \n"
                "categorical_vars = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'LoanAmount'] \n"
                " \n"
                "for var in categorical_vars: \n"
                "   if var != 'LoanAmount': \n"
                "       df_missing_var.append(var) \n"
                "       df_missing_imp.append(df[var].mode()[0]) \n"
                "       df_missing_type.append('Categorical') \n"
                "   else: \n"
                "       df_missing_var.append(var) \n"
                "       df_missing_imp.append(round(df[var].mean(),2)) \n"
                "       df_missing_type.append('Numerical') \n"
                " \n"
                "df_missing_final = pd.DataFrame({'feature': df_train_missing_var, 'impute using value': df_train_missing_imp, 'type (Cat = Mode; Num = Mean)': df_train_missing_type})")

        df_missing_var = []
        df_missing_imp = []
        df_missing_type = []

        categorical_vars = ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term", "Credit_History", "LoanAmount"]

        for var in categorical_vars:
            if var != "LoanAmount":
                df_missing_var.append(var)
                df_missing_imp.append(df[var].mode()[0])
                df_missing_type.append("Categorical")
            else:
                df_missing_var.append(var)
                df_missing_imp.append(round(df[var].mean(),2))
                df_missing_type.append("Numerical")

        df_missing_final = pd.DataFrame({'feature': df_missing_var, 'impute using value': df_missing_imp, 'type (Cat = Mode; Num = Mean)': df_missing_type})

        st.write(df_missing_final)

        st.caption("**Before imputation, :orange[**let's first create a copy of our original DataFrame**] so as if we encounter errors in the future, \n"
                 "we won't disrupt our original dataset**")
        
        st.code("df_copy = df_train.copy()")
        df_copy = df.copy()

        st.caption("**Below is the summary (in dictionary form) of the features and the values that we're going to use for imputation**")

        st.code("missing_val_map = dict(df_train_missing_final.drop(columns=['type (Cat = Mode; Num = Mean)']).values)")
        missing_val_map = dict(df_missing_final.drop(columns=["type (Cat = Mode; Num = Mean)"]).values)
        st.write(missing_val_map)

        for k, v in missing_val_map.items():
            for col in df_copy.columns:
                if k == col:
                    df_copy[col] = df_copy[col].fillna(v)
                else:
                    pass

        st.code("for k, v in missing_val_map.items(): \n"
                "   for col in df_copy.columns: \n"
                "       if k == col: \n"
                "           df_copy[col] = df_copy[col].fillna(v) \n"
                "       else: \n"
                "           pass")

        st.write(df_copy)

        st.caption("**Let's check again to make sure that we don't have NAs in our datasets anymore**")

        st.code("check_train = df_copy.isnull().sum().to_frame('NAs')")
        check_train = df_copy.isnull().sum().to_frame('NAs')
        st.write(check_train)

        st.markdown("""---""")

        st.write("**Outliers**: There are tons of outliers under each of the numerical features. :orange[**Let's further look into each one of them**]")

        st.caption(":green[**ApplicantIncome**]")
        st.code("fig = plt.figure(figsize=(18, 2)) \n"
                "sns.boxplot(x=df_copy['ApplicantIncome'], palette='mako') \n"
                "st.pyplot(fig) \n"
                " \n"
                "df_copy.sort_values(by=['ApplicantIncome'], ascending=False).head(5)")        
        fig = plt.figure(figsize=(18, 2))
        sns.boxplot(x=df_copy["ApplicantIncome"], palette="mako")
        st.pyplot(fig)

        st.write(df_copy.sort_values(by=["ApplicantIncome"], ascending=False).head(5))
        st.caption("➡ **Most of the outliers are in the higher end of the range. Though, based on my observation, none of these values \n"
                   "are out of the ordinary. Having a deeper look, we could see that :orange[**most of the applicants with the high income have many dependents (could mean \n"
                   "that they're older and working in a much higher role in their company), are graduated, not self-employed (working in corporate), and living in a \n"
                   "urban/suburban location**]**.")

        st.caption(":green[**CoapplicantIncome**]")
        st.code("fig = plt.figure(figsize=(18, 2)) \n"
                "sns.boxplot(x=df_copy['CoapplicantIncome'], palette='mako') \n"
                "st.pyplot(fig) \n"
                " \n"
                "df_copy.sort_values(by=['CoapplicantIncome'], ascending=False).head(5)")        
        fig = plt.figure(figsize=(18, 2))
        sns.boxplot(x=df_copy["CoapplicantIncome"], palette="mako")
        st.pyplot(fig)

        st.write(df_copy.sort_values(by=["CoapplicantIncome"], ascending=False).head(5))
        st.caption("➡ **Again, for CoapplicantIncome, the values don't seem to look out of the ordinary. What we could interpret is :orange[**most of the people with \n"
                   "high CoapplicantIncome have low ApplicantIncome, meaning they surely need an independent with a higher income for their loans to be approved**]**.")

        st.caption(":green[**LoanAmount**]")
        st.code("fig = plt.figure(figsize=(18, 2)) \n"
                "sns.boxplot(x=df_copy['LoanAmount'], palette='mako') \n"
                "st.pyplot(fig) \n"
                " \n"
                "df_copy.sort_values(by=['LoanAmount'], ascending=False).head(5)")        
        fig = plt.figure(figsize=(18, 2))
        sns.boxplot(x=df_copy["LoanAmount"], palette="mako")
        st.pyplot(fig)

        st.write(df_copy.sort_values(by=["LoanAmount"], ascending=False).head(5))
        st.caption("➡ **:orange[**There's really nothing much we could say for this since anyone, regardless of background and status, can avail a loan**]. As far as \n"
                 "observing the numbers, there's nothing out of the ordinary**.")

        st.markdown("""---""")        

        st.write("**Change to Indicator variables**: As mentioned earlier, since we have categorical variables, :orange[**it's best to convert them to indicator variables**] \n"
                 "so we could use them in building our model (An Indicator variable is a numerical variable that encodes categorical information)")

        st.caption("**Before proceeding, we should convert :orange[**Loan_Amount_Term**] first to a categorical variable since it shows grouping, hence considered a categorical variable**")
        st.code("df_copy['Loan_Amount_Term'] = df_copy['Loan_Amount_Term'].astype(str)")
        df_copy["Loan_Amount_Term"] = df_copy["Loan_Amount_Term"].astype(str)

        st.caption("**Now let's use the function :orange[**get_dummies**] to change numerical variables to indicator variables**")
        st.code("df_copy_dummies = pd.get_dummies(data=df_copy, drop_first=True)")
        df_copy_final = pd.get_dummies(data=df_copy, drop_first=True)
        st.write(df_copy_final)
        st.caption("**As you can see, we specifically used the parameter :orange[**drop_first**] in our code; this is to completely eliminate the first category per feature breakdown. \n"
                 "For example, we can only see one column for Gender, which is :orange[**Gender_Male**], even though we should also have another one for :orange[**Gender_Female**]. \n"
                 "This is to completely eliminate the possibility of multicollinearity within our features**.")

        st.markdown("""---""")

    #MODELLING
    with st.container():
        st.subheader("📍 Modelling")
        st.subheader("1️⃣ Model Fitting (Decision Tree vs. Random Forest)")
        st.write("This is where we fit our training sets into our models and adjust the parameters for better accuracy")
        st.write("Let's first split our dependent and independent variables into two separate DataFrames")

        with st.container():
            st.code("X = pd.DataFrame(df_copy.drop('Loan_Status', axis=1)) \n"
                    "y = pd.DataFrame(df_copy['Loan_Status'])")
            X = pd.DataFrame(df_copy_final.drop("Loan_Status_Y", axis=1))
            y = pd.DataFrame(df_copy_final["Loan_Status_Y"])

        st.write("Then, we perform :orange[**Train-Test Split**] to split our data into training and testing sets for both dependent and independent variables")

        with st.container():
            st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2)")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)

        st.write("Now that our data is ready, then we fit our training set to our model")

        col3, col4 = st.columns(2)

        with col3:
            st.caption(":green[**Decision Tree**]")
            st.code("dtr = DecisionTreeClassifier(random_state=2) \n"
                    "dtr.fit(X_train, y_train)")
            dtr = DecisionTreeClassifier(random_state=2)
            dtr.fit(X_train, y_train)

        with col4:
            st.caption(":green[**Random Forest**]")
            st.code("rfc = RandomForestClassifier(random_state=2) \n"
                    "rfc.fit(X_train, y_train)")
            rfc = RandomForestClassifier(random_state=2)
            rfc.fit(X_train, y_train)

        st.markdown("""---""")

        #Overfit check
        st.subheader("✋ Check for overfitting")
        st.write("Before proceeding, let's first check for overfitting by using the :orange[**score()**] function for both training and test sets")

        col5, col6 = st.columns(2)

        with col5:
            st.caption(":green[**Decision Tree**]")
            st.code("score_train_dtr = dtr.score(X_train, y_train) \n"
                    "score_test_dtr = dtr.score(X_test, y_test)")

            score_train_dtr = dtr.score(X_train, y_train)
            score_test_dtr = dtr.score(X_test, y_test)
            st.write("**Train set score:**", score_train_dtr)
            st.write("**Test set score:**", score_test_dtr)

        with col6:
            st.caption(":green[**Random Forest**]")
            st.code("score_train_rfc = rfc.score(X_train, y_train) \n"
                    "score_test_rfc = rfc.score(X_test, y_test)")

            score_train_rfc = rfc.score(X_train, y_train)
            score_test_rfc = rfc.score(X_test, y_test)
            st.write("**Train set score:**", score_train_rfc)
            st.write("**Test set score:**", score_test_rfc)

        st.caption("➡ **As you can see, we got perfect scores in our training sets, but their differences against the test sets are very big, so there's overfitting that exists. It would be \n"
                   "best to perform :orange[**Hyperparameter Tuning**] to really bring out the best parameters that would produce the models with the highest accuracy and \n"
                   "remove overfitting**.")
        
        st.markdown("""---""")
        
        #GridSearchCV
        st.subheader("⚡️ Hyperparameter Tuning with RandomizedSearchCV")
        st.write("This is where we introduce the function :orange[**RandomizedSearchCV**] to iterate and test different combinations of parameters. Some of its differences with **GridSearchCV** \n"
                 "is that it randomly selects from a list of parameters, so it doesn't guarantee the parameter combination with the best accuracy, but it certainly is more efficient and faster \n" 
                 "since it doesn't check all sets of parameters.")

        col7, col8 = st.columns(2)
        
        with col7:
            st.write(":green[**Decision Tree**]")

        with col8: 
            st.write(":green[**Random Forest**]")

        st.caption("**Let's set parameter grids for both of our algorithms containing different hyperparameters, which we'll be used by RandomizedSearchCV**")

        col9, col10 = st.columns(2)

        with col9:
            dtr_param_grid = {
                'max_depth': [2, 4, 8, 16, 32],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'min_samples_split': [2, 4, 6, 8, 10]
            }

            st.code("dtr_param_grid = { \n"
                    "   'max_depth': [2, 4, 8, 16, 32], \n"
                    "   'criterion': ['gini', 'entropy'], \n"
                    "   'max_features': ['auto', 'sqrt', 'log2'], \n"
                    "   'min_samples_split': [2, 4, 6, 8, 10] \n"
            "}")

        with col10:
            rfc_param_grid = {
                'n_estimators': [5, 10, 50, 100, 250],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [2, 4, 8, 16, 32],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [1, 2],
                'bootstrap': [True, False]
            }

            st.code("rfc_param_grid = { \n"
                    "   'n_estimators': [5, 10, 50, 100, 250], \n"
                    "   'max_features': ['auto', 'sqrt'], \n"
                    "   'max_depth': [2, 4, 8, 16, 32], \n"
                    "   'min_samples_split': [2, 4, 6, 8, 10], \n"
                    "   'min_samples_leaf': [1, 2], \n"
                    "   'bootstrap': [True, False] \n"
            "}")

        st.caption("**Now we run our RandomizedSearchCV**")

        col11, col12 = st.columns(2)

        with col11:
            # dtr_rand_search = RandomizedSearchCV(estimator=dtr, 
            #                                      param_distributions=dtr_param_grid, 
            #                                      cv=10,
            #                                      n_iter=10,
            #                                      n_jobs=-1,
            #                                      scoring='accuracy')
            
            st.code("dtr_rand_search = RandomizedSearchCV(estimator=dtr, \n"
                    "                                     param_distributions=dtr_param_grid, \n"
                    "                                     cv=10, \n"
                    "                                     n_iter=10, \n"
                    "                                     n_jobs=-1, \n"
                    "                                     scoring='accuracy')) \n")

        with col12:
            # rfc_rand_search = RandomizedSearchCV(estimator=rfc, 
            #                                      param_distributions=rfc_param_grid, 
            #                                      cv=10,
            #                                      n_iter=10,
            #                                      n_jobs=-1,
            #                                      scoring='accuracy')
            
            st.code("rfc_rand_search = RandomizedSearchCV(estimator=rfc, \n"
                    "                                     param_distributions=rfc_param_grid, \n"
                    "                                     cv=10, \n"
                    "                                     n_iter=10, \n"
                    "                                     n_jobs=-1, \n"
                    "                                     scoring='accuracy')) \n")


        st.caption("**These are the best parameters generated by the RandomizedSearchCV**")

        col13, col14 = st.columns(2)

        with col13:
            dtr_best_params = {
                'min_samples_split': 6,
                'max_features': 'auto',
                'max_depth': 4,
                'criterion': 'gini'
            }

            st.code("dtr_best_params.best_params_")
            st.write(dtr_best_params)

        with col14:
            rfc_best_params = {
                'n_estimators': 100,
                'min_samples_split': 10,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'max_depth': 4,
                'bootstrap': False,
            }

            st.code("rfc_best_params.best_params_")
            st.write(rfc_best_params)

        st.caption("**We now use the values in the best parameters dictionary in our classifier and fit our datasets**")

        col15, col16 = st.columns(2)

        with col15:
            dtr_rand_search = DecisionTreeClassifier(min_samples_split=6, max_features='auto', max_depth=4, criterion='gini', random_state=2)
            dtr_rand_search.fit(X_train, y_train)
            st.code("dtr_rand_search = DecisionTreeClassifier(min_samples_split=6, max_features='auto', max_depth=4, criterion='gini', random_state=2) \n"
                    "dtr_rand_search.fit(X_train, y_train)")

            st.write("**Training set accuracy**: ", dtr_rand_search.score(X_train, y_train))
            st.write("**Test set accuracy**: ", dtr_rand_search.score(X_test, y_test))

        with col16:
            rfc_rand_search = RandomForestClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=1, max_features='sqrt', max_depth=4, bootstrap=False, random_state=2)
            rfc_rand_search.fit(X_train, y_train)
            st.code("rfc_rand_search = RandomForestClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=1, max_features='sqrt', max_depth=4, bootstrap=False, random_state=2) \n"
                    "rfc_rand_search.fit(X_train, y_train)")

            st.write("**Training set accuracy**: ", rfc_rand_search.score(X_train, y_train))
            st.write("**Test set accuracy**: ", rfc_rand_search.score(X_test, y_test))

        st.caption("➡ **As you can see, :orange[**we have now solved the problem of overfitting!**] The differences in the accuracies of our training and test sets for both of the models \n"
                   "have drastically reduced**.")

        st.markdown("""---""")

        #Model Evaluation
        st.subheader("2️⃣ Model Evaluation")
        st.write("Let's now predict and evaluate our model using a :orange[**Confusion Matrix and a Classification Report**]")
        st.write("**Confusion Matrix**")

        col17, col18 = st.columns(2)

        with col17:
            st.write(":green[**Decision Tree**]")

        with col18: 
            st.write(":green[**Random Forest**]")

        col19, col20 = st.columns(2)

        with col19:
            st.code("dtr_y_pred = dtr_rand_search.predict(X_test) \n"
                    " \n"
                    "dtr_cm = confusion_matrix(y_test, dtr_y_pred) \n"
                    "dtr_cm_norm = dtr_cm.astype('float') / dtr_cm.sum(axis=1)[:, np.newaxis] \n"
                    "dtr_fig = plt.figure(figsize=(10, 5)) \n"
                    "dtr_ax = plt.subplot() \n"
                    "sns.heatmap(dtr_cm_norm, cmap='mako', annot=True, ax=dtr_ax) \n"
                    " \n"
                    "dtr_ax.set_xlabel('True labels') \n"
                    "dtr_ax.set_ylabel('Predicted labels') \n"
                    "dtr_ax.xaxis.set_ticklabels(['Loan_Status_Y = 0', 'Loan_Status_Y = 1']) \n"
                    "dtr_ax.yaxis.set_ticklabels(['Loan_Status_Y = 0', 'Loan_Status_Y = 1']) \n")

            dtr_y_pred = dtr_rand_search.predict(X_test)

            dtr_cm = confusion_matrix(y_test, dtr_y_pred)
            dtr_cm_norm = dtr_cm.astype('float') / dtr_cm.sum(axis=1)[:, np.newaxis]
            dtr_fig = plt.figure(figsize=(10, 5))
            dtr_ax = plt.subplot()
            sns.heatmap(dtr_cm_norm, cmap="mako", annot=True, ax=dtr_ax)

            dtr_ax.set_xlabel('True labels')
            dtr_ax.set_ylabel('Predicted labels')
            dtr_ax.xaxis.set_ticklabels(['Loan_Status_Y = 0', 'Loan_Status_Y = 1'])
            dtr_ax.yaxis.set_ticklabels(['Loan_Status_Y = 0', 'Loan_Status_Y = 1'])

            st.pyplot(dtr_fig)

        with col20:
            st.code("rfc_y_pred = dtr_rand_search.predict(X_test) \n"
                    " \n"
                    "rfc_cm = confusion_matrix(y_test, rfc_y_pred) \n"
                    "rfc_cm_norm = rfc_cm.astype('float') / rfc_cm.sum(axis=1)[:, np.newaxis] \n"
                    "rfc_fig = plt.figure(figsize=(10, 5)) \n"
                    "rfc_ax = plt.subplot() \n"
                    "sns.heatmap(rfc_cm_norm, cmap='mako', annot=True, ax=rfc_ax) \n"
                    " \n"
                    "rfc_ax.set_xlabel('True labels') \n"
                    "rfc_ax.set_ylabel('Predicted labels') \n"
                    "rfc_ax.xaxis.set_ticklabels(['Loan_Status_Y = 0', 'Loan_Status_Y = 1']) \n"
                    "rfc_ax.yaxis.set_ticklabels(['Loan_Status_Y = 0', 'Loan_Status_Y = 1']) \n")

            rfc_y_pred = rfc_rand_search.predict(X_test)

            rfc_cm = confusion_matrix(y_test, rfc_y_pred)
            rfc_cm_norm = rfc_cm.astype('float') / rfc_cm.sum(axis=1)[:, np.newaxis]
            rfc_fig = plt.figure(figsize=(10, 5))
            rfc_ax = plt.subplot()
            sns.heatmap(rfc_cm_norm, cmap="mako", annot=True, ax=rfc_ax)

            rfc_ax.set_xlabel('True labels')
            rfc_ax.set_ylabel('Predicted labels')
            rfc_ax.xaxis.set_ticklabels(['Loan_Status_Y = 0', 'Loan_Status_Y = 1'])
            rfc_ax.yaxis.set_ticklabels(['Loan_Status_Y = 0', 'Loan_Status_Y = 1'])

            st.pyplot(rfc_fig)
        
        st.caption("➡ **For both models, we can absolutely see high predictive performances when it comes to predicting :orange[**Loan_Status_Y = 1 at 96% \n"
                   "and 97%**] for Decision Tree and Random Forest, respectively, but performed poorer when it comes to predicting :orange[**Loan_Status_Y = 0, both at 40%**]**")

        st.write("**Classification Report**")

        col21, col22 = st.columns(2)

        with col21:
            st.write(":green[**Decision Tree**]")

        with col22: 
            st.write(":green[**Random Forest**]")

        col23, col24 = st.columns(2)

        with col23:
            dtr_report = classification_report(y_test, dtr_y_pred, output_dict=True)
            dtr_report_df = pd.DataFrame(dtr_report).transpose()
            st.write(dtr_report_df)

        with col24:
            rfc_report = classification_report(y_test, rfc_y_pred, output_dict=True)
            rfc_report_df = pd.DataFrame(rfc_report).transpose()
            st.write(rfc_report_df)

        st.caption("➡ **Almost in all of the metrics, the :orange[**Random Forest model performed better than the Decision Tree**], but only slightly.** \n")

        st.write("**ROC and AUROC scores**")

        r_probs = [0 for _ in range(len(y_test))]
        dtr_probs = dtr_rand_search.predict_proba(X_test)
        rfc_probs = rfc_rand_search.predict_proba(X_test)

        dtr_probs = dtr_probs[:, 1]
        rfc_probs = rfc_probs[:, 1]

        r_auc = roc_auc_score(y_test, r_probs)
        dtr_r_auc = roc_auc_score(y_test, dtr_probs)
        rfc_r_auc = roc_auc_score(y_test, rfc_probs)

        r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
        dtr_fpr, dtr_tpr, _ = roc_curve(y_test, dtr_probs)
        rfc_fpr, rfc_tpr, _ = roc_curve(y_test, rfc_probs)

        fig = plt.figure(figsize=(12, 6))
        plt.plot(r_fpr, r_tpr, linestyle="--", label="Random Prediction: AUROC = %.3f" % r_auc)
        plt.plot(dtr_fpr, dtr_tpr, marker=".", label="Decision Tree: AUROC = %.3f" % dtr_r_auc)
        plt.plot(rfc_fpr, rfc_tpr, marker=".", label="Random Forest: AUROC = %.3f" % rfc_r_auc)

        plt.title("ROC Plot")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend()
        plt.show()

        st.pyplot(fig)

        st.caption("➡ **It is pretty clear that the Random Forest model performed better than the Decision Tree model :orange[**because \n"
                   "of higher AUROC score**].**")
        
        st.markdown("""---""")

        #Feature Importance
        st.subheader("3️⃣ Feature Importance")
        st.write("Feature Importance is very important for us to have a better understanding of how our model was influenced by the set \n"
                 "of features that we used. That way, we'll learn which of them are the best predictors.")
        
        feature_importances = pd.Series(dtr_rand_search.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        feature_importances_df = pd.DataFrame(feature_importances).reset_index().rename(columns={"index":"features", 0:"score"})
        feature_importances_df["score"] = feature_importances_df["score"].round(2)

        fig = plt.figure(figsize=(16, 5))
        y_pos = range(len(feature_importances_df["features"]))
        plt.xticks(y_pos, feature_importances_df["features"], rotation=90)

        ax = sns.barplot(data=feature_importances_df.head(10), x="features", y="score")
        ax.bar_label(ax.containers[0])

        st.pyplot(fig)

        st.caption("➡ **It seems that :orange[**Credit_History**] was the biggest predictor in whether a person would get a loan or not.**")

        st.markdown("""---""")

        st.caption("**Important references:**")

        st.caption("https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74")
        st.caption("https://www.section.io/engineering-education/hyperparmeter-tuning/")
        st.caption("https://www.section.io/engineering-education/hyperparmeter-tuning/")
        st.caption("https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56")
        st.caption("https://www.datacamp.com/tutorial/random-forests-classifier-python")
        st.caption("https://www.statology.org/sklearn-classification-report/")
        st.caption("https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance")
        st.caption("https://muthu.co/understanding-the-classification-report-in-sklearn/#:~:text=A%20Classification%20report%20is%20used,classification%20report%20as%20shown%20below.")
        st.caption("https://medium.com/analytics-vidhya/hyper-parameter-tuning-gridsearchcv-vs-randomizedsearchcv-499862e3ca5")
        st.caption("https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397")

        
