# Step 1: Setup
from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

from pathlib import Path
parent = Path(__file__).resolve().parent
resources_folder = parent / "resources"
docs_dir = resources_folder / "groundwork_docs"
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
docs = SimpleDirectoryReader(docs_dir).load_data()





# Step 2: Load the Documents
print("Num Docs loaded:", len(docs))
for i, doc in enumerate(docs):
    print(f"Doc {i+1}: {doc.metadata["file_name"]}")
print()




# Step 3: Build the Index and Query Engine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini", temperature=0.2) # The default model gpt-3.5-turbo was deprecated on October 23rd 2026
Settings.llm = llm

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

print("Index built successfully. Ready to answer questions.")





# Step 4: Query the Assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print("Question:", q)
    response = query_engine.query(q)
    print()
    print("Answer:", response)

    print()

    #top_node = response.source_nodes[0]
    top_node = response.source_nodes[0]
    print("Filename:", top_node.metadata["file_name"])
    print(f"Similarity Score: {top_node.score:.4f}")
    print("First 200 Chars:", top_node.node.get_content()[:200])
    print()
    print("-="*20 + "-")
    print()
"""
For each response, the assistant was confident and accurate in it's responses. However, I was suprised because the top similarity
scored response for some of the questions was not the document that had the answer. For example, the questioon for "What are 
Groundwork's hours on weekends" returned the our_story.txt file as the highest similarity score, even though it did not mention
the store hours.
"""





# Step 5: Find a Failure
q = "Will Groundwork have plans to open a store in Florida?"
print("Question:", q)
response = query_engine.query(q)
print()
print("Answer:", response)

print()


for this_node in response.source_nodes:
    print("Filename:", this_node.metadata["file_name"])
    print(f"Similarity Score: {this_node.score:.4f}")
    print("First 200 Chars:", this_node.node.get_content()[:200])
    print()

print("-="*20 + "-")
print()

"""
I asked it a question that cannot be answered because the answer is not contained within the text. The question asked was 
as follows: "Will Groundwork have plans to open a store in Florida?". It responded with the following: "There is no 
information available regarding plans for Groundwork to open a store in Florida".

The model answered by saying there is no information to answer the text. It did not guess or assume, it stated there was no
information to answer the question and stopped.

The tone of the model did not change much. It was confident in it's answer about not having any information to answer the
original question. 

Something I would add to the system is that if it can't find an answer or is not confident, then the model could respond
by providing a company email if the user wants further inquire about the question that couldn't be answered.
"""



# Step 6: Reflection
"""
1.  My LlamaIndex implementation took less than 10 lines of code compared to the 40 lines for a manual implementation of
    even a simple keyword-based RAG. The LlamaIndex framework not only makes it much simpler and quicker, it also improves
    the result by performing Semantic RAG instead of keywword-based RAG.

2.  Another implementation for this could be quickly ensuring any newly added information for a company or it's policies is
    not conflicting and coincides with the existing structure and policies in place. 

3.  RAG cannot prevent a hallucination of information. If the model manages to retrieve all the information, even if it is,
    correct, it might generate wrong information and provide that to the user as confidently correct.
"""