import pymupdf
from sentence_transformers import SentenceTransformer
import faiss
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#Check file extension
def check_file_extension(file):
    return "." in file and file.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

#Convert PDF into blocks
def get_text_blocks(file):
    document = pymupdf.open(file)
    text_blocks = []

    for page in document:
        blocks = page.get_text("blocks")
        for b in blocks:
            text_blocks.append(b[4])

    return text_blocks

def create_faiss_index(text_blocks):
    embeddings = model.encode(text_blocks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def search_manual(query, text_blocks, index, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=top_k)

    best_matches = [text_blocks[i] for i in I[0]]
    return "\n\n".join(best_matches)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "" or not check_file_extension(file.filename):
        return jsonify({"error": "Invalid file"})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    text_blocks = get_text_blocks(file_path)
    index, _ = create_faiss_index(text_blocks)

    return jsonify({"message": "File uploaded successfully", "text_blocks": text_blocks})

@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.json
    query = data.get("query")
    text_blocks = data.get("text_blocks")

    if not query or not text_blocks:
        return jsonify({"error": "Missing data"})

    index, _ = create_faiss_index(text_blocks)
    answer = search_manual(query, text_blocks, index)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

