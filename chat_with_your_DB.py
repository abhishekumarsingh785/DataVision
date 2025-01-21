#!/usr/bin/env python
# coding: utf-8

# In[72]:


import os
import re
import streamlit as st
import sqlite3
from langchain.agents import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import AIMessage
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.sql_database import SQLDatabase
from langchain.tools import Tool,StructuredTool
#from langchain_groq import ChatGroq  # Importing the LLaMA model via Groq
import joblib
from langchain_ollama import ChatOllama


# In[ ]:





# In[73]:


#from pydantic import BaseModel


# In[ ]:





# In[74]:


import matplotlib.pyplot as plt


# In[75]:


import os
import dotenv
dotenv.load_dotenv()


# In[76]:


regression_model = joblib.load('sales_regression_model.pkl')


# In[77]:


#llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, api_key=os.environ["GROQ_API_KEY"])


# In[78]:


llm = ChatOllama(model = "llama3.1:8b", base_url = "http://localhost:11434", temperature=0,)


# In[79]:


#llm = ChatOpenAI(model="gpt-4o", temperature=0)


# In[81]:


conn = sqlite3.connect('Chinook_Sqlite.sqlite')
database = SQLDatabase.from_uri("sqlite:///Chinook_Sqlite.sqlite")


# In[80]:


EXAMPLES = """
Q: Total sales per customer
SQL: SELECT CustomerId, SUM(Total) FROM Invoice GROUP BY CustomerId;

Q: Average invoice total by country
SQL: SELECT BillingCountry, AVG(Total) FROM Invoice GROUP BY BillingCountry;

Q: List all tracks along with their album title and artist name
SQL: SELECT Track.Name AS TrackName, Album.Title AS AlbumTitle, Artist.Name AS ArtistName
     FROM Track
     JOIN Album ON Track.AlbumId = Album.AlbumId
     JOIN Artist ON Album.ArtistId = Artist.ArtistId;

Q: Find the total number of tracks in each playlist
SQL: SELECT Playlist.Name AS PlaylistName, COUNT(PlaylistTrack.TrackId) AS NumberOfTracks
     FROM Playlist
     JOIN PlaylistTrack ON Playlist.PlaylistId = PlaylistTrack.PlaylistId
     GROUP BY Playlist.Name;

Q: Show the names of customers and the total amount they have spent
SQL: SELECT Customer.FirstName || ' ' || Customer.LastName AS CustomerName, SUM(Invoice.Total) AS TotalSpent
     FROM Customer
     JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId
     GROUP BY Customer.CustomerId;

Q: Find the most popular genre based on the number of tracks sold
SQL: SELECT Genre.Name AS GenreName, COUNT(InvoiceLine.TrackId) AS NumberOfTracksSold
     FROM Genre
     JOIN Track ON Genre.GenreId = Track.GenreId
     JOIN InvoiceLine ON Track.TrackId = InvoiceLine.TrackId
     GROUP BY Genre.Name
     ORDER BY NumberOfTracksSold DESC;
"""


# In[82]:


synonym_mapping = {
    # Customers Table
    "customer": "CustomerId",
    "client id": "CustomerId",
    "customer number": "CustomerId",
    "client": "CustomerId",
    "cust id": "CustomerId",
    "first name": "FirstName",
    "given name": "FirstName",
    "customer name": "FirstName",
    "client name": "FirstName",
    "last name": "LastName",
    "surname": "LastName",
    "family name": "LastName",
    "company": "Company",
    "business": "Company",
    "organization": "Company",
    "address": "Address",
    "location": "Address",
    "city": "City",
    "town": "City",
    "state": "State",
    "region": "State",
    "province": "State",
    "country": "Country",
    "nation": "Country",
    "postal code": "PostalCode",
    "zip code": "PostalCode",
    "postcode": "PostalCode",
    "phone": "Phone",
    "telephone": "Phone",
    "phone number": "Phone",
    "email": "Email",
    "email address": "Email",
    "contact email": "Email",
    
    # Invoices Table
    "invoice id": "InvoiceId",
    "bill id": "InvoiceId",
    "invoice number": "InvoiceId",
    "bill number": "InvoiceId",
    "order id": "InvoiceId",
    "order number": "InvoiceId",
    "invoice date": "InvoiceDate",
    "bill date": "InvoiceDate",
    "order date": "InvoiceDate",
    "purchase date": "InvoiceDate",
    "billing address": "BillingAddress",
    "invoice address": "BillingAddress",
    "order address": "BillingAddress",
    "billing city": "BillingCity",
    "invoice city": "BillingCity",
    "order city": "BillingCity",
    "billing state": "BillingState",
    "invoice state": "BillingState",
    "order state": "BillingState",
    "billing country": "BillingCountry",
    "invoice country": "BillingCountry",
    "order country": "BillingCountry",
    "billing postal code": "BillingPostalCode",
    "invoice postal code": "BillingPostalCode",
    "order postal code": "BillingPostalCode",
    "billing zip code": "BillingPostalCode",
    "total": "Total",
    "invoice total": "Total",
    "bill total": "Total",
    "order total": "Total",
    "amount": "Total",
    
    # Artists Table
    "artist id": "ArtistId",
    "band id": "ArtistId",
    "musician id": "ArtistId",
    "artist name": "Name",
    "band name": "Name",
    "musician name": "Name",
    "group name": "Name",
    
    # Albums Table
    "album id": "AlbumId",
    "record id": "AlbumId",
    "cd id": "AlbumId",
    "album title": "Title",
    "record title": "Title",
    "cd title": "Title",
    
    # Tracks Table
    "track id": "TrackId",
    "song id": "TrackId",
    "tune id": "TrackId",
    "music id": "TrackId",
    "track name": "Name",
    "song name": "Name",
    "tune name": "Name",
    "music title": "Name",
    "composer": "Composer",
    "writer": "Composer",
    "song writer": "Composer",
    
    # MediaTypes Table
    "media type id": "MediaTypeId",
    "format id": "MediaTypeId",
    "media type": "Name",
    "format": "Name",
    
    # Playlists Table
    "playlist id": "PlaylistId",
    "list id": "PlaylistId",
    "mix id": "PlaylistId",
    "playlist name": "Name",
    "mix name": "Name",
    "list name": "Name",
    
    # InvoiceLines Table
    "invoice line id": "InvoiceLineId",
    "line item id": "InvoiceLineId",
    "bill line id": "InvoiceLineId",
    "order line id": "InvoiceLineId",
    "unit price": "UnitPrice",
    "line price": "UnitPrice",
    "item price": "UnitPrice",
    "quantity": "Quantity",
    "number": "Quantity",
    "amount": "Quantity"
}


# In[83]:


# Function to preprocess user input and replace synonyms
def preprocess_user_input(user_input, synonym_mapping):
    user_input_lower = user_input.lower()
    for alias, actual_name in synonym_mapping.items():
        user_input_lower = user_input_lower.replace(alias.lower(), actual_name)
    return user_input_lower


# In[84]:


# Enhanced PII Masking Function
def mask_pii_data(text):
    # Mask emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '[REDACTED]', text)
    
    # Mask phone numbers (example formats: 123-456-7890, (123) 456-7890, 123 456 7890, 123.456.7890)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED]', text)
    
    # Mask addresses (simple example for street numbers and names)
    text = re.sub(r'\b\d{1,5}\s\w+\s\w+\b', '[REDACTED]', text)
    
    # Mask names (first and last)
    text = re.sub(r'\b([A-Z][a-z]*\s[A-Z][a-z]*)\b', '[REDACTED]', text)
    
    return text


# In[85]:


sql_examples = [
    {"input": "List all customers.", "query": "SELECT * FROM Customer;"},
    {"input": "What is the total sales per customer?", "query": "SELECT CustomerId, SUM(Total) AS TotalSales FROM Invoice GROUP BY CustomerId;"},
    {"input": "Show the names of customers and their invoices.", "query": "SELECT Customer.FirstName, Customer.LastName, Invoice.InvoiceId, Invoice.Total FROM Customer JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId;"},
    {"input": "Find all customers from Germany.", "query": "SELECT * FROM Customer WHERE Country = 'Germany';"},
    {"input": "List the top 5 customers by total spending.", "query": "SELECT CustomerId, SUM(Total) AS TotalSpent FROM Invoice GROUP BY CustomerId ORDER BY TotalSpent DESC LIMIT 5;"},
    {"input": "Find albums that have more than 10 tracks.", "query": "SELECT AlbumId, COUNT(TrackId) AS NumberOfTracks FROM Track GROUP BY AlbumId HAVING COUNT(TrackId) > 10;"},
    {"input": "Show all invoices from the last year.", "query": "SELECT * FROM Invoice WHERE InvoiceDate >= DATE('now', '-1 year');"},
    {"input": "Show the full names of customers.", "query": "SELECT FirstName || ' ' || LastName AS FullName FROM Customer;"},
    {"input": "Calculate the average invoice total.", "query": "SELECT AVG(Total) AS AverageTotal FROM Invoice;"},
    {"input": "What is the distribution of tracks across different genres?", "query": "SELECT Genre.Name AS GenreName, COUNT(Track.TrackId) AS NumberOfTracks FROM Genre JOIN Track ON Genre.GenreId = Track.GenreId GROUP BY Genre.Name;"},
    {"input": "List all tracks with their album and artist names.", "query": "SELECT Track.Name AS TrackName, Album.Title AS AlbumTitle, Artist.Name AS ArtistName FROM Track JOIN Album ON Track.AlbumId = Album.AlbumId JOIN Artist ON Album.ArtistId = Artist.ArtistId;"},
    {"input": "Count the number of customers with and without email addresses.", "query": "SELECT SUM(CASE WHEN Email IS NOT NULL THEN 1 ELSE 0 END) AS WithEmail, SUM(CASE WHEN Email IS NULL THEN 1 ELSE 0 END) AS WithoutEmail FROM Customer;"},
    {"input": "Sales over the years", "query":"SELECT strftime('%Y', i.InvoiceDate) AS Year, SUM(il.UnitPrice * il.Quantity) AS TotalSales FROM Invoice i JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId GROUP BY Year ORDER BY Year;"},

]


# In[86]:


example_selector = SemanticSimilarityExampleSelector.from_examples(
    sql_examples,
    OpenAIEmbeddings(model = "text-embedding-3-small"),
    FAISS,
    k=2,
    input_keys=["input"],
)


# In[88]:


PREFIX = """
You are a SQL expert. You have access to a SQLite database.
Identify which tables can be used to answer the user's question and write and execute a SQL query accordingly.
Given an input question, create a syntactically correct SQL query to run against the dataset customer_profiles, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the col umns from a specific table; only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the information returned by these tools to construct your final answer.
You MUST double-check your query before executing it. If you get an error while executing a query, rewrite the query and try again.DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
If the question does not seem related to the database, just return "I don't know" as the answer. 
If the user asks for a visualization of the results, generate a complete python code for this purpose, that can be directly executed. Always include import matplotlib.pyplot as plt.
"""

SUFFIX = """Begin!
{chat_history}
Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    prefix=PREFIX,
    suffix="",
    input_variables=["input", "top_k"],
    example_separator="\n\n",
)

messages = [
    SystemMessagePromptTemplate(prompt=few_shot_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=SUFFIX),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate.from_messages(messages)


# In[87]:


python_repl = PythonREPL()


# In[89]:


def execute_python_code(python_code):
    exec_globals = {}
    exec_locals = {}
    try:
        exec(python_code, exec_globals, exec_locals)
        
        # Check if any figures were created
        if plt.get_fignums():
            fig = plt.gcf()  # Get the current figure
            st.session_state.plot = fig  # Store the plot in session state
            plt.close(fig)  # Close the plot to prevent it from showing multiple times
        
        return exec_locals
    except Exception as e:
        return {"error": str(e)}


# In[ ]:

regression_model = joblib.load('sales_regression_model.pkl')

def predict_sales(year):
    try:
        prediction = regression_model.predict([[year]])[0]
        return f"Predicted sales for the year {year}: ${prediction:.2f}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"



# In[90]:


def sql_agent_tools():
    tools = [
        StructuredTool.from_function(
            func=mask_pii_data,
            name="mask_pii_data",
            description="Masks PII data in the input text.",
        ),
    ]
    return tools


# In[91]:


extra_tools = []

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, input_key="input"
)
# Create the agent executor
agent_executor = create_sql_agent(
    llm=llm,
    db=database,
    verbose=True,
    top_k=10,
    prompt=prompt,
    extra_tools=extra_tools,
    input_variables=["input", "agent_scratchpad", "chat_history"],
    agent_type="openai-tools",
    agent_executor_kwargs={"handle_parsing_errors": True, "memory": memory},
)


# In[92]:


def extract_and_execute_code(response):
    try:
        # Extract the Python code using a regular expression
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            python_code = code_match.group(1)
            print(f"Extracted Python Code:\n{python_code}")
            
            # Execute the extracted Python code manually
            exec_locals = execute_python_code(python_code)
            return exec_locals
        
        print("No Python code found in the LLM response.")
        return "No Python code found."
    
    except Exception as e:
        
        print(f"Error during execution: {e}")
        return str(e)

# Function to run the agent with enhanced logging and synonym mapping
def run_agent(input_text):
    try:
        processed_input = preprocess_user_input(input_text, synonym_mapping)

        # Check for prediction request
        if "predict" in processed_input.lower():
            match = re.search(r'\b\d{4}\b', processed_input)
            if match:
                year = int(match.group(0))
                prediction_response = predict_sales(year)
                st.session_state.response = prediction_response
                st.write(f"Prediction Response: {prediction_response}")
                return prediction_response
            else:
                no_year_msg = "No valid year found for prediction."
                st.session_state.response = no_year_msg
                st.write(no_year_msg)
                return no_year_msg

        # Run the SQL agent
        response = agent_executor.run(input=processed_input)

        # Remove Python code for display purposes
        response_without_code = re.sub(r'To visualize this.*', '', response, flags=re.DOTALL).strip()
        response_without_code = re.sub(r'Here is the Python code to.*', '', response_without_code, flags=re.DOTALL).strip()
        
        st.session_state.response = response_without_code
        st.write(f"Agent Response: {response_without_code}")
        
        # Extract and execute any Python code if present
        if isinstance(response, str):
            exec_locals = extract_and_execute_code(response)
            return exec_locals
        else:
            print(f"Agent Response: {response}")
            return response
    
    except Exception as e:
        print(f"Error during agent execution: {e}")
        return str(e)

# In[93]:


#response = run_agent("what is the yearly sales trend, show this in a line chart")


# In[94]:


# Streamlit UI
st.title("DataVision 🤖")

# Ensure session state has 'history', 'plot', and 'response' keys
if "history" not in st.session_state:
    st.session_state.history = []

if "plot" not in st.session_state:
    st.session_state.plot = None

if "response" not in st.session_state:
    st.session_state.response = None

user_input = st.text_input("Ask your question:")

if st.button("Run Query"):
    if user_input:
        with st.spinner("Processing..."):
            # Clear session state on new input
            st.session_state.history = []
            st.session_state.plot = None
            st.session_state.response = None

            run_agent(user_input)  # Use the updated run_agent function here
            
            st.experimental_rerun()
    else:
        st.error("Please enter a question.")

# Display the stored response and plot if they exist
if st.session_state.response:
    st.write(st.session_state.response)

if st.session_state.plot:
    st.pyplot(st.session_state.plot)

for message in st.session_state.history:
    if isinstance(message, str):
        st.write(message)

# In[ ]:




