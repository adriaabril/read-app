# READ-APP

## Overview
This application allows users to upload a PDF file (such as a board game manual) and ask questions about its content.
The app processes the document and retrieves the most relevant answer using NLP techniques and FAISS for efficient search.

## Features

- Upload a PDF document
- Extract and process text efficiently
- Search for answers using FAISS and embeddings
- Provide relevant responses based on user queries

## Installation
### Requirements
- Python 3.8+
- Required libraries:

```
pip install pymupdf faiss-cpu sentence-transformers streamlit

```