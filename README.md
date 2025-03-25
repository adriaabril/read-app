# READ-APP

## Overview
This application allows users to upload a PDF file (such as a board game manual) and ask questions about its content.
The app processes the document and retrieves the most relevant answer using NLP techniques and FAISS for efficient search.

## Features

- Upload PDFs: Users can upload PDF documents via a web interface.

- Automatic Text Extraction: The app extracts and processes text from the uploaded PDFs.

- Fast Search with FAISS: Uses FAISS and Sentence Transformers to find relevant answers.

- Web Interface: A user-friendly Flask-based interface for uploading PDFs and querying information.

## Installation
### Requirements
- Python 3.8+
- Required libraries:

```
pip install pymupdf faiss-cpu sentence-transformers flask werkzeug
```

## Running the Application
- Clone the repository:

```
git clone https://github.com/yourusername/read-app.git
cd read-app
```

- Set up a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

- Install dependencies:

```
pip install -r requirements.txt
```

- Run the Flask app:

```
python main.py
```

- Open a web browser and go to:

```
http://127.0.0.1:5000/
```

## Usage
### Uploading a PDF

- Navigate to the Upload section of the web interface.

- Select and upload a PDF file.

- The system will process and extract the document's text.

### Querying the PDF

- Enter a query related to the uploaded document.

- The system will search and return the most relevant answer based on embeddings and FAISS indexing.