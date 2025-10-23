from sentence_transformers import SentenceTransformer

# Load the model (downloads automatically the first time)
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Model loaded: {model}")
print(f"Max sequence length: {model.max_seq_length}")

model.encode(text)
