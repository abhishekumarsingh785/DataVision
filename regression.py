from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline




import sqlite3

# Path to the downloaded Chinook database file
database_path = 'Chinook_Sqlite.sqlite'

try:
    # Connect to the Chinook database
    conn = sqlite3.connect(database_path)

    # Fetch the list of tables
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = conn.execute(tables_query).fetchall()

    # Display the tables
    print("Tables in the database:")
    for table in tables:
        print(table[0])

except sqlite3.DatabaseError as e:
    print(f"Database error: {e}")
finally:
    # Close the connection
    if conn:
        conn.close()




import sqlite3
import pandas as pd

# Connect to the Chinook database
conn = sqlite3.connect('Chinook_Sqlite.sqlite')

# SQL query to get total sales per year
query = """
SELECT 
    ar.Name AS ArtistName, 
    SUM(il.Quantity) AS TotalTracksSold 
FROM 
    Artist ar
JOIN 
    Album al ON ar.ArtistId = al.ArtistId 
JOIN 
    Track t ON al.AlbumId = t.AlbumId 
JOIN 
    InvoiceLine il ON t.TrackId = il.TrackId 
GROUP BY 
    ar.Name 
ORDER BY 
    TotalTracksSold DESC 
LIMIT 1;

"""

# Load the data into a DataFrame
sales_data = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

# Display the data
print(sales_data)




model = LinearRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=2, scoring='r2')


#just CV = 5


X = sales_data[['Year']]  # Features: Year
y = sales_data['TotalSales']  # Target: Total Sales

# Initialize the linear regression model
model = LinearRegression()

# Perform cross-validation to evaluate the model
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Fit the model on the entire dataset
model.fit(X, y)

# Predict on the same data
predictions = model.predict(X)

# Calculate R^2 score on the entire dataset
r2 = r2_score(y, predictions)

X = sales_data[['Year']]  # Features: Year
y = sales_data['TotalSales']  # Target: Total Sales

# Initialize the linear regression model
model = LinearRegression()

# Initialize Leave-One-Out Cross-Validation
loo = LeaveOneOut()

# Arrays to store predictions and actual values
predictions = []
actuals = []

# Perform LOOCV
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict the test set
    y_pred = model.predict(X_test)
    
    # Store the results
    predictions.append(y_pred[0])
    actuals.append(y_test.iloc[0])

# Calculate the R^2 score for the LOOCV predictions
r2 = r2_score(actuals, predictions)

# Fit the model on the entire dataset
model.fit(X, y)


X = sales_data[['Year']]  # Features: Year
y = sales_data['TotalSales']  # Target: Total Sales

# Create polynomial features with degree 2
polynomial_features = PolynomialFeatures(degree=2)
model = make_pipeline(polynomial_features, LinearRegression())

# Fit the model on the data
model.fit(X, y)

# Predict on the same data
predictions = model.predict(X)

# Calculate the R² score
r2 = r2_score(y, predictions)



joblib.dump(model, 'sales_regression_model.pkl')


