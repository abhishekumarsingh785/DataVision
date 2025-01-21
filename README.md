# DataVision: Effortless Data Insights and Predictions 📊✨

Welcome to **DataVision**, an intuitive and powerful application designed to help businesses unlock the full potential of their data. Whether you're a seasoned data expert or a business leader, DataVision simplifies querying, visualizing, and forecasting data, empowering smarter decisions and strategic growth.

![Demo Video](https://drive.google.com/file/d/1zB0wvKRHI3LcY9Aza7RhdfG4jGsV87z0/view?usp=sharing)

---

## 🚀 **Features**

### ✨ Dynamic Data Querying
- Connected to the **Chinook dataset** stored in a **SQLite database**.
- Ask natural language questions to get accurate, actionable insights.

### 📊 Interactive Visualizations
- Not just text answers—DataVision creates **visual plots** based on your queries, making complex data easy to understand.

### 🔮 Prediction Capabilities
- Powered by a **pre-trained regression model**, DataVision predicts future values based on user input, adding foresight to your data analysis.

### 🧠 Advanced Language Model Integration
- Built using the **LLaMA 3.1 8B model** via **Ollama** for robust and precise language understanding.

### 🔒 Data Security
- Runs locally on your laptop, ensuring **data privacy** and **security**.

### 💡 Enhanced Understanding with FAISS
- Utilizes **FAISS** for **few-shot learning**, ensuring accurate SQL query generation by understanding complex relationships in your data.

### 🔄 Synonym Mapper
- Handles different terminologies for tables and columns seamlessly, so you don’t need to worry about exact field names.

---

## 🎥 **Demo Video**
See DataVision in action! Watch the demo: [DataVision Demo Video](https://drive.google.com/file/d/1zB0wvKRHI3LcY9Aza7RhdfG4jGsV87z0/view?usp=sharing)

---

## 🛠 **Technical Overview**
- **Database**: Chinook SQLite database, modeled after the iTunes store, with interconnected tables such as albums, artists, customers, invoices, and tracks.
- **Language Model**: LLaMA 3.1 8B via Ollama for natural language understanding.
- **Few-shot Learning**: FAISS integration for accurate query formulation.
- **Prediction**: Trained regression model to forecast sales trends.
- **Visualization**: Auto-generated Python scripts for interactive plots.
- **Synonym Mapping**: Effortlessly maps user inputs to database fields.

---

## 📝 **How It Works**
1. **Query the Data**: Type your question in plain English.
2. **Insightful Answers**: Receive text-based insights or visual representations.
3. **Predict the Future**: Use the built-in model to forecast trends.
4. **Enjoy Privacy**: All computations happen locally on your device.

---

## 🌟 **Tailorable Solution**
While the demo leverages the Chinook dataset, DataVision can connect to any database, making it adaptable for various use cases.

---
## 📂 **Files in the Repository**

- `chat_with_your_DB.py`: Main application file for querying and visualizing data.
- `regression.py`: Script to train and save the regression model used for predictions.
- `sales_regression_model.pkl`: Pre-trained regression model for predicting sales trends.
- `Chinook_Sqlite.sqlite`: SQLite database resembling the iTunes store with interconnected tables.
