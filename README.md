# Chat with Websites and PDFs

This project allows users to interact with content from websites or PDFs in a conversational format. It uses advanced natural language processing models and vector databases to retrieve and generate responses based on user queries.

## Features

- **Chat with Documents:** Interact with content from websites or uploaded PDFs.
- **Vector Search:** Use PostgreSQL and PGVector for efficient semantic search.
- **Customizable LLMs:** Integrate Hugging Face models for embedding generation and conversational AI.
- **Streamlit Interface:** Simple and user-friendly interface for interaction.

## Tech Stack

- **Programming Language:** Python
- **Web Framework:** Streamlit
- **Database:** PostgreSQL with PGVector extension
- **NLP Models:** Hugging Face Transformers
- **Environment Management:** dotenv

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up PostgreSQL with PGVector

1. Install PostgreSQL.
2. Install the PGVector extension. Refer to the [PGVector documentation](https://github.com/pgvector/pgvector) for installation instructions.
3. Create a database and configure the `.env` file with your PostgreSQL credentials.

### 4. Configure Environment Variables

Create a `.env` file in the project directory and add the following variables:

```plaintext
DB_USER=your_postgres_username
DB_PASSWORD=your_postgres_password
DB_HOST=your_postgres_host
DB_PORT=your_postgres_port
DB_NAME=your_postgres_database_name
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

### 5. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the app in your browser at `http://localhost:8501`.

---

## How to Use

1. **Load Content:**
   - Enter a website URL or upload a PDF file via the sidebar.
   - The content will be processed and stored in a PostgreSQL vector database.

2. **Chat:**
   - Type your queries in the chat input field.
   - The application retrieves relevant content and generates a response based on the conversation history.

3. **Clear Chat:**
   - Use the "Clear Chat History" button in the sidebar to reset the chat and vector store.

---

## Directory Structure

```
📁 Project Root
├── app.py                # Main application script
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── .gitignore            # Ignored files for Git
└── README.md             # Documentation
```

---

## Dependencies

Install the following Python packages (specified in `requirements.txt`):

- `streamlit`
- `langchain-core`
- `langchain-community`
- `langchain-postgres`
- `langchain-huggingface`
- `huggingface_hub`
- `psycopg2`
- `pypdf`
- `python-dotenv`

---

## Example Usage

### 1. Chat with a Website
- Enter a valid website URL in the sidebar.
- Example: `https://example.com`
- The bot will load content and respond to your queries.

### 2. Chat with a PDF
- Upload a PDF file using the file uploader.
- Example: Upload a document with FAQs, and the bot will generate responses based on the content.

---

## Troubleshooting

### Common Issues

- **Database Connection Error:**
  Ensure PostgreSQL is running and the credentials in the `.env` file are correct.

- **Model Loading Issue:**
  Verify your Hugging Face API key and ensure you have access to the specified models.

- **PDF Parsing Errors:**
  Ensure the uploaded PDF is not corrupted and contains readable text.

## Acknowledgments

- **LangChain** for providing modular building blocks for LLM applications.
- **Hugging Face** for their state-of-the-art NLP models.
- **PGVector** for efficient vector search in PostgreSQL.