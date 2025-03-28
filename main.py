import pymupdf
from sentence_transformers import SentenceTransformer
import re
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
    structured_blocks = {
        "titles": [],
        "text": [],
        "tables": []
    }
    for page in document:
            blocks = page.get_text("blocks")
            for b in blocks:
                text = b[4].strip()

                if len(text) < 5:  # Avoid fragments too short
                    continue

                # Possible title
                if text.isupper() or len(text) < 50:
                    structured_blocks["titles"].append(text)

                # Possible tables
                elif any(char.isdigit() for char in text) and "\n" in text:
                    structured_blocks["tables"].append(text)

                else:
                    structured_blocks["text"].append(text)

    return structured_blocks

#Extract lists of rules from text
def extract_rules(text_blocks):
    rules = []
    current_rule = ""

    for text in text_blocks:
        lines = text.split("\n")
        for line in lines:
            if re.match(r"^\s*[\•\-0-9]+\.", line):
                if current_rule:
                    rules.append(current_rule.strip())
                current_rule = line
            else:
                current_rule += " " + line

    if current_rule:
        rules.append(current_rule.strip())

    return rules

## Create FAISS index with categorized sections
def create_faiss_index(structured_blocks):
    all_texts = structured_blocks["titles"] + structured_blocks["text"] + structured_blocks["tables"]
    
    if not all_texts:
        print("Error: No texts to index!")
        return None, []
    
    embeddings = model.encode(all_texts, convert_to_numpy=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, all_texts

def search_manual(query, structured_blocks, index, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k=top_k)

    all_texts = structured_blocks["titles"] + structured_blocks["text"] + structured_blocks["tables"]

    valid_results = [i for i in I[0] if 0 <= i < len(all_texts)]
    
    if not valid_results:
        return "No relevant information found in the manual."

    results = [all_texts[i] for i in valid_results]
    context = "\n\n".join(results)
    
    return f"According to the manual, here's the relevant information:\n\n{context}"


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

    structured_blocks = get_text_blocks(file_path)
    extracted_rules = extract_rules(structured_blocks["text"])
    index, _ = create_faiss_index(structured_blocks)

    print("Structured blocks:", structured_blocks)
    print("Extracted rules:", extracted_rules)

    return jsonify({
        "message": "File uploaded successfully",
        "structured_blocks": structured_blocks,
        "rules": extracted_rules
    })

@app.route("/query", methods=["POST"])
def query_pdf():
    data = request.json
    query = data.get("query")
    structured_blocks = data.get("structured_blocks")

    if not query or not structured_blocks:
        return jsonify({"error": "Missing data"})

    all_texts = structured_blocks["titles"] + structured_blocks["text"] + structured_blocks["tables"]
    index, _ = create_faiss_index({"titles": [], "text": all_texts, "tables": []})

    if not structured_blocks or "text" not in structured_blocks:
        return jsonify({"error": "Invalid structured_blocks format"})

    answer = search_manual(query, structured_blocks, index)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

