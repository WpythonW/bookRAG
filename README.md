# BookRAG: Intelligent Book Interaction Application

## 🚀 Project Overview

BookRAG is an intelligent application that allows users to interact with book content using advanced RAG (Retrieval-Augmented Generation) techniques.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.10+
- pip
- venv

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/WpythonW/bookRAG.git
cd bookRAG
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and fill in the following details:

```
PROXY_HOST=your_proxy_host
PROXY_PORT=your_proxy_port
PROXY_USER=your_proxy_username
PROXY_PASSWORD=your_proxy_password
GEMINI_API_KEY=your_gemini_api_key
```

## 🚀 Running the Application

```bash
streamlit run app.py
```

## 🔐 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys and proxy credentials confidential

## 📦 Dependencies

The project uses the following key technologies:
- Streamlit
- Gemini API
- Python RAG techniques

## 📄 License

MIT, Apache 2.0