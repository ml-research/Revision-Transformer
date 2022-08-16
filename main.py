from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
from datasets import load_from_disk
import sys

sys.path.append('/workspace/repositories/norms')

model_name = "facebook/rag-sequence-nq"
# model_name = "facebook/rag-token-nq"

dataset = load_from_disk(f'/workspace/repositories/norms/processed_norms_train')
dataset.load_faiss_index("embeddings", f"/workspace/repositories/norms/processed_norms_index_train.faiss")

# Easy way to load the model
retriever = RagRetriever.from_pretrained(model_name, index_name="custom", indexed_dataset=dataset)
model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)
model.cuda()
tokenizer = RagTokenizer.from_pretrained(model_name)

# query
question = "Is it good to celebrate with your family?"
# question = "Q: Is it moral to kill time? A:"
# question = "Q: Should I troll people? A:"
input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"].cuda()
generated = model.generate(input_ids)
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(question)
print(generated_string)
