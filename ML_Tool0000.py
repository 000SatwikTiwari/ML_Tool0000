import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import ttest_ind, ttest_rel, f_oneway, chi2_contingency
from itertools import combinations
from pymongo import MongoClient
import gridfs

client = MongoClient("mongodb://satwiksudhanshtiwari:Satwik2021@ac-0afcv37-shard-00-00.8hns6ba.mongodb.net:27017,ac-0afcv37-shard-00-01.8hns6ba.mongodb.net:27017,ac-0afcv37-shard-00-02.8hns6ba.mongodb.net:27017/?ssl=true&replicaSet=atlas-3m9dyh-shard-0&authSource=admin&retryWrites=true&w=majority&appName=Cluster0")
db = client['ml_tool']
collection = db['datasets']
fs = gridfs.GridFS(db)

def save_to_mongodb(file, file_name):
    existing_dataset = collection.find_one({"filename": file_name})
    if existing_dataset:
        st.warning(f"A dataset with the title '{file_name}' already exists. Upload skipped.")
    else:
        file_id = fs.put(file, filename=file_name)
        collection.insert_one({"filename": file_name, "file_id": file_id})
        st.success(f"Dataset '{file_name}' uploaded successfully.")

def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            save_to_mongodb(uploaded_file.getvalue(), uploaded_file.name)
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            try:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                save_to_mongodb(uploaded_file.getvalue(), uploaded_file.name)
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                st.error("The file encoding is not supported or the file is empty. Please upload a valid CSV file with UTF-8 or ISO-8859-1 encoding.")
                return None
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a non-empty CSV file.")
            return None
        return df
    return None

def get_datasets():
    datasets = collection.find()
    return {dataset['filename']: dataset['file_id'] for dataset in datasets}

def preprocess_data(df):
    st.info("🔍 **Purpose:** This function helps you explore your dataset, identify missing values, and choose how to handle them before training the machine learning model.")
    
    # Dataset Preview
    st.info("📊 **Step 1: Dataset Preview** – Displays the first few rows to give an overview of the data structure.")
    st.write("## Dataset Preview")
    st.write(df.head())
    
    # Statistical Overview
    st.info("📈 **Step 2: Statistical Overview** – Shows summary statistics (mean, std, min, max, etc.) for numerical features.")
    st.write("## Statistical Overview")
    st.write(df.describe())

    # Missing Values Detection
    st.info("❗ **Step 3: Missing Values Detection** – Displays the number of missing entries in each column.")
    st.write("## Missing Values")
    st.write(df.isnull().sum())
    for column in df.columns:
        if df[column].isnull().sum() > 0:

            st.info(f"🛠️ **Step 4: Handle Missing Values in `{column}`** – Choose a strategy to handle null values in this column.")
            
            st.write(f"Column: {column}")
            
            action = st.selectbox(
                f"Choose action for {column}", 
                options=["None", "Remove Rows", "Replace with Mean", "Replace with Median", "Replace with Mode"], 
                key=column
            )

            if action == "Remove Rows":
                st.info(f"🧹 Removing rows with missing values in `{column}`.")
                df = df.dropna(subset=[column])
            elif action == "Replace with Mean":
                st.info(f"➕ Replacing missing values in `{column}` with the **mean**.")
                df[column].fillna(df[column].mean(), inplace=True)
            elif action == "Replace with Median":
                st.info(f"➕ Replacing missing values in `{column}` with the **median**.")
                df[column].fillna(df[column].median(), inplace=True)
            elif action == "Replace with Mode":
                st.info(f"➕ Replacing missing values in `{column}` with the **mode**.")
                df[column].fillna(df[column].mode()[0], inplace=True)
                
    return df

def convert_data_types(df):
    st.info("🔍 **Purpose:** This function helps you convert each column to an appropriate data type, ensuring that the data is in the right format for machine learning.")
    
    # Data Type Conversion Overview
    st.info("⚙️ **Step 1: Data Type Overview** – Displays the current data types of each column.")
    st.write("## Data Type Conversion")
    updated_df = df.copy()

    for column in df.columns:
        st.write(f"Column: {column}, Current Data Type: {df[column].dtype}")
        new_type = st.selectbox(f"Convert {column} to", options=["None", "int", "float", "str"], key=f"convert_{column}")
        
        if new_type != "None":
            st.info(f"🔄 **Step 2: Convert `{column}` to {new_type}** – Converts the column to the selected data type.")
            try:
                if new_type == 'str':
                    updated_df[column] = updated_df[column].astype(str)
                else:
                    updated_df[column] = updated_df[column].astype(new_type)
            except ValueError:
                st.warning(f"Conversion of {column} to {new_type} failed. Please choose a compatible data type.")
    
    st.write("## Final Dataset Columns with Data Types")
    st.write(updated_df.dtypes)
    return updated_df

def show_standard_deviation(df):
    st.info("🔍 **Purpose:** This function calculates and displays the standard deviation of numerical columns to understand the spread or variability of the data.")
    
    # Standard Deviation Calculation
    st.info("📊 **Step 1: Select Standard Deviation Calculation Method** – Choose how you want to calculate the standard deviation: Mean, Median, or Mode.")
    st.write("## Standard Deviation of Columns")
    method = st.selectbox("Calculate Standard Deviation with respect to", ["Mean", "Median", "Mode"])
    std_devs = {}
    
    st.info("📈 **Step 2: Standard Deviation Calculation** – The function calculates the standard deviation of each numerical column based on the selected method.")
    
    for column in df.select_dtypes(include=[np.number]).columns:
        if method == "Mean":
            std_dev = np.std(df[column])
        elif method == "Median":
            std_dev = np.std(df[column] - df[column].median())
        elif method == "Mode":
            std_dev = np.std(df[column] - df[column].mode()[0])
        std_devs[column] = std_dev
    sorted_std_devs = dict(sorted(std_devs.items(), key=lambda item: item[1]))
    st.write(sorted_std_devs)

def encode_data(df):
    st.info("🔍 **Purpose:** This function encodes categorical features into numerical values so that machine learning models can process them.")
    
    # Label Encoding Overview
    st.info("🛠️ **Step 1: Label Encoding** – Convert categorical columns into numerical labels for machine learning models.")
    label_encoders = {}
    
    for column in df.select_dtypes(include=['object']).columns:
        st.write(f"Encoding column: `{column}`")
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
        
    return df, label_encoders

def train_models(X, y, task):
    
    st.info("🔍 **Purpose:** This function trains multiple machine learning models and returns the trained models for evaluation.")
    
    # Model Selection
    st.info("⚙️ **Step 1: Select Models** – Based on the task (Classification or Regression), different models are chosen.")
    models = {}
    
    if task == 'Classification':
        models['Logistic Regression'] = LogisticRegression()
        models['Decision Tree'] = DecisionTreeClassifier()
        models['Random Forest'] = RandomForestClassifier()
    else:
        models['Linear Regression'] = LinearRegression()
        models['Decision Tree'] = DecisionTreeRegressor()
        models['Random Forest'] = RandomForestRegressor()
        
    trained_models = {}
    
    st.info("🏋️‍♀️ **Step 2: Train Models** – The function trains each selected model on the provided data (X, y).")
    
    for name, model in models.items():
        model.fit(X, y)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(trained_models, X_test, y_test, task):
    st.info("🔍 **Purpose:** This function evaluates the performance of the trained models using test data.")
    
    # Model Evaluation
    st.info("⚖️ **Step 1: Evaluate Models** – Evaluate each trained model using accuracy (for classification) or RMSE (for regression).")
    evaluations = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        if task == 'Classification':
            evaluations[name] = accuracy_score(y_test, y_pred)
        else:
            evaluations[name] = mean_squared_error(y_test, y_pred, squared=False)
            
    return evaluations

def visualize_data(df):
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Visualization Purpose
    st.info("🔍 **Purpose:** This function visualizes the dataset with various types of charts to help understand the relationships and distributions within the data.")
    
    # Data Visualization Step
    st.info("📊 **Step 1: Select Chart Type** – Choose the type of chart you would like to generate based on the data columns.")
    columns = df.columns.tolist()
    if columns:
        chart_type = st.selectbox("Select chart type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Pie Chart"])
        if chart_type in ["Histogram", "Bar Plot", "Pie Chart"]:
            x_axis = st.selectbox("Select column", columns)
        else:
            x_axis = st.selectbox("Select X-axis column", columns)
            y_axis = st.selectbox("Select Y-axis column", columns)
        plt.figure(figsize=(10, 6))
        if chart_type == "Scatter Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.scatterplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Line Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.lineplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Bar Plot":
            st.write(f"{chart_type} of {x_axis}")
            sns.countplot(data=df, x=x_axis)
        elif chart_type == "Histogram":
            st.write(f"{chart_type} of {x_axis}")
            sns.histplot(df[x_axis], kde=True)
        elif chart_type == "Box Plot":
            st.write(f"{chart_type} of {x_axis} and {y_axis}")
            sns.boxplot(data=df, x=x_axis, y=y_axis)
        elif chart_type == "Pie Chart":
            st.write(f"Pie Chart of {x_axis}")
            df[x_axis].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
            plt.ylabel('')
        st.pyplot(plt)
    else:
        st.error("The CSV file does not contain any columns.")



def main():
    fixed_string = "Developer -> Satwik Tiwari\REC BIJNOR(IT)\n from Prayagraj."

    st.write(fixed_string)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Preprocessing", "Visualization", "Model Training", "Predict", "Datasets"])

    if page == "Upload":
        st.title("Upload and Preprocess Data")
        data = load_data()
        

    elif page == "Preprocessing":
        st.title("Data Preprocessing")
        st.session_state.clear()
        data = load_data()
        if data is not None:
            data = preprocess_data(data)
            data = convert_data_types(data)

            st.write("## Correlation Heatmap")
            correlation = None
            try:
                correlation = data.corr()
                plt.figure(figsize=(12, 8))
                sns.heatmap(correlation, annot=True, cmap='coolwarm')
                st.pyplot(plt)
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {e}")

            if correlation is not None and len(data.columns) > 20:
                st.write("## Select Columns for Correlation Analysis")
                input_columns = st.multiselect("Select Input Columns", options=data.columns)
                target_column = st.selectbox("Select Target Column", options=data.columns)
                if input_columns and target_column:
                    numeric_columns = data.select_dtypes(include=[np.number]).columns
                    if target_column in numeric_columns:
                        correlations = correlation[target_column].abs().sort_values(ascending=False)
                        top_columns = correlations.head(9).index.tolist()
                        top_columns.remove(target_column)
                        st.write(f"Top 8 columns correlated with {target_column}: {top_columns}")
                        selected_columns = list(set(input_columns).intersection(top_columns + [target_column]))
                        data = data[selected_columns]
                st.write("## Final Dataset After Selecting Top Correlated Columns")
                st.write(data.head())
            else:
                drop_columns = st.multiselect("Select columns to drop", options=data.columns)
                if st.button("Drop Selected Columns"):
                    data.drop(columns=drop_columns, inplace=True)
                    
                    st.write("Selected columns dropped successfully.")

            st.write("## Download Modified Dataset")
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Modified Dataset", data=csv, file_name="modified_dataset.csv", mime='text/csv')
            st.session_state.clear()

            
    elif page == "Visualization":
        st.title("CSV Data Visualization")
        data = load_data()
        if data is not None:
            visualize_data(data)

    # [everything remains unchanged above]

    elif page == "Model Training":
        st.session_state.clear()
        st.title("Model Training")
        data = load_data()
        if data is not None:
            st.title("Model Training")
            
            data = preprocess_data(data)
            input_columns = st.multiselect("Select Input Columns", options=data.columns)
            target_column = st.selectbox("Select Target Column", options=data.columns)
            if input_columns and target_column:
                X = data[input_columns]
                y = data[target_column]
                X, _ = encode_data(X)
                y, label_encoder = encode_data(pd.DataFrame(y))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                task = st.radio("Select Task Type", ('Classification', 'Regression'))
                trained_models = train_models(X_train, y_train.values.ravel(), task)
                evaluations = evaluate_models(trained_models, X_test, y_test.values.ravel(), task)
                st.write("## Model Performance")
                st.write(evaluations)
                selected_model = st.selectbox("Select Model", options=list(trained_models.keys()))
                st.session_state["selected_model"] = trained_models[selected_model]
                st.session_state["input_columns"] = input_columns
                st.session_state["label_encoder"] = label_encoder.get(target_column) if task == 'Classification' else None

# [everything else remains unchanged below]


    elif page == "Predict":
        st.title("Make Predictions")
        if "selected_model" in st.session_state:
            model = st.session_state["selected_model"]
            input_columns = st.session_state["input_columns"]
            label_encoder = st.session_state.get("label_encoder", None)
            user_input = {}
            for col in input_columns:
                user_input[col] = st.text_input(f"Input {col}")
            if st.button("Predict"):
                try:
                    user_input_df = pd.DataFrame([user_input])
                    user_input_df, _ = encode_data(user_input_df)
                    prediction = model.predict(user_input_df)
                    if label_encoder:
                        prediction = label_encoder.inverse_transform(prediction)
                    st.write(f"Prediction: {prediction}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.error("Please train a model first in the 'Model Training' page.")
    
    elif page == "Datasets":
        st.title("Available Datasets")
        datasets = get_datasets()
        
        st.write("## Search Datasets")
        search_term = st.text_input("Search by name")
        filtered_datasets = {name: file_id for name, file_id in datasets.items() if search_term.lower() in name.lower()}
        
        st.write("## Dataset List")
        for name, file_id in filtered_datasets.items():
            if st.button(name):
                file_data = fs.get(file_id).read()
                st.download_button(f"Download {name}", file_data, file_name=name)

if __name__ == '__main__':
    main()
