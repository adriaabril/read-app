import pymupdf
from sentence_transformers import SentenceTransformer
import faiss

filename = "RegicideRulesA4.pdf"
model = SentenceTransformer("all-MiniLM-L6-v2")

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


def main():

    text_blocks = get_text_blocks(filename)
    index, embeddings = create_faiss_index(text_blocks)

    query = "How does the Jester work?"
    answer = search_manual(query, text_blocks, index)

    print(answer)


main()
