from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")





# -=-=- RAG Concepts -=-=-
# Concepts Q1
"""
Scenario A: RAG
    The data that is being accessed is constantly being updated and, since it's about legal policy, needs to be grounded in truth. Therefore,
    using a RAG approach allows them to ensure that any data retrieved is directly from the text source in their company policy.

Scenario B: Fine-tuning
    The company wants to essentially retrain the LLM using their provided examples so that it can adopt a specific style. 

Scenario C: Prompt Engineering
    The data analyst only needs to review a single 2 page report. Therefore, the report can be uploaded and reviewed in a single prompt
    without having to completely train the LLM on new data to provide specific responses or styles. 
"""

# Concepts Q2
"""
A confidently wrong answer is much more harmful than one that says "I am not sure" because the latter informs the user that the answer may 
have wrong information which prompts them to check if the information is indeed factual. A confidently wrong answer does not do that, and 
instead, may have the user use it thinking it is correct when it is indeed wrong.

An example of this could be an AI confidently telling you that "Everyone goes 5mph over the posted speed limit so it will be fine" when,
in reality, any amount of speed above the posted speed limit can get you pulled over.
"""

# Concepts Q3
"""
steps = [
    "Extract text from source documents",       # Take and store the source text
    "Split text into chunks",                   # Split the text into smaller sections for individual processing
    "Convert text chunks into embeddings",      # Convert each section of text into a number that defines it's meaning
    "Receive the user's query",                 # Get the user's question to the LLM
    "Embed the user's query",                   # Turn the user's question into a number
    "Retrieve the most relevant chunks",        # compare the number from the user to the numbers of the text and get the most similar
    "Inject retrieved chunks into the prompt",  # Answer the questions using the provided data chunks
    "Generate a response from the LLM",         # Create a response to send back to the user
]    
"""





# -=-=- Keyword RAG -=-=-
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Q1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

answer = simple_keyword_retrieval(query, documents, verbose=True)
print(answer)
print()
# The selected document was "loyalty.txt" because each of the documents except for "menu.txt" had an overlap score of 1. However, 
# loyalty.txt" is the last one in the dictionary and because the scores is sorted in reverse order, it becomes the first one and
# is subsequently chosen.  





# Keyword Q2
query = "Do you have anything without caffeine?"
answer = simple_keyword_retrieval(query, documents, verbose=True)
print(answer)
print()
# With this retrieval, there was "no overlapping keywords found." This means that in each of the documents, it didn't find any of 
# the words like "anything", "caffeine", "do", "have", or "without". As such, each of the documents scored 0 on the overlap and
# none was retrieved. What could make this better is by adding Semantic RAG, which searches for the meaning of the words rather
# than the exact overlap of words.





# Keyword Q3

# I predict that it will also return no document because there is no overlap between any of the query words and document text.
query = "How do I sign up for rewards?"
answer = simple_keyword_retrieval(query, documents, verbose=True)
print(answer)
print()
print("-="*20 + "-")
print()
# My prediction was correct. I saw no overlap between words in the documents and the retrieval confirmed that.





# -=-=- Semantic RAG -=-=-
# Semantic Q1
"""
1.  A Vector Embedding is given to a piece of text as a number which stores it's meaning. It is compared to other texts 
    by checking if their numbers are similar. If the text is similar, then it will have a similar vector embedding.

2.  The chunk that has a cosine similarity of 0.85 is more relevant to the query as it has a higher cosine similarity.
    Furthermore, a cosine similarity of 1 is the highest possible, so 0.85 is a very similar match. The 0.30 is a smaller
    cosine similarity so it is not scored as relevant as the other text.

3.  Semantic searches look for the text's meaning, not each individual keyword. The meaning behind "Yo what's up" and 
    "Hi, how are you" is identical, and yet no words match. However, with semantic searching, it could find these phrases
    as similar and relevant.
"""

# Semantic Q2
"""
| Feature                    | Keyword RAG                       | Semantic RAG      |
|----------------------------|-----------------------------------|-------------------|
| What is compared?          | Exact word overlap                | Word Meaning      |
| What is retrieved?         | Full document                     | Text Chunk        |
| Can it handle synonyms?    | No                                | yes               |
| Storage format             | Plain text dictionary             | Vector Embedding  |
| Relevance score            | Number of overlapping keywords    | Cosine Similarity |
"""





# -=-=- LLamaIndex -=-=-

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o-mini", temperature=0.2) # The default model gpt-3.5-turbo was deprecated on October 23rd 2026
Settings.llm = llm # The default model gpt-3.5-turbo was deprecated on October 23rd 2026

from pathlib import Path
parent = Path(__file__).resolve().parent
resources_folder = parent / "resources"
docs_dir = resources_folder / "brightleaf_pdfs"

docs = SimpleDirectoryReader(docs_dir).load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)


# LlamaIndex Q1
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print("Question:", q)
    response = query_engine.query(q)
    print("Answer:", response)

    print()

    for this_node in response.source_nodes:
        print("Node ID:", this_node.node.node_id)
        print(f"Similarity Score: {this_node.score:.4f}")
        print("First 150 Chars:", this_node.node.get_content()[:150])
        print()

"""
For the 1st query:
1. Yes, the first retrieved chunk is relevant with a score of 0.758, however the second and third are less relevant, even though
they have a similarity score of over 0.5

2. The model sounds confident and does not use any phrases that show uncertainty.

3. Yes, it retrieved information about the EcoVolt Energy partnerships and the mission statement, however it did not talk about
them in it's answer.


For the 2nd query:
1.  Yes the first document has a similarity score of 0.680 and talks about the security policies at BrightLeaf.

2.  Once again, it has the same level of confidence, relaying teh information directly without any hesitation in its response.

3.  Similarly to the 1st query, it retrieved information from the partnerships and employee benefits documents but didn't use any
    information from those documents in the answer.
"""

print()
print("-="*20 + "-")
print()

# LlamaIndex Q2
query = "What employee benefits does BrightLeaf offer?"

# 1st run:
query_engine = index.as_query_engine(similarity_top_k=5)

print("Question:", query)
response = query_engine.query(query)
print("Answer:", response)
for this_node in response.source_nodes:
    print("Node ID:", this_node.node.node_id)
    print(f"Similarity Score: {this_node.score:.4f}")
    print("First 150 Chars:", this_node.node.get_content()[:150])
    print()


# 2nd run:
query_engine = index.as_query_engine(similarity_top_k=1)

print("Question:", query)
response = query_engine.query(query)
print("Answer:", response)
for this_node in response.source_nodes:
    print("Node ID:", this_node.node.node_id)
    print(f"Similarity Score: {this_node.score:.4f}")
    print("First 150 Chars:", this_node.node.get_content()[:150])
    print()

# The response changed slightly in terms of format only as the 1st run had a paragraphed response while the 2nd run
# had a numbered list response. Both however, had the same information. More context is not always better, as it can
# confuse the LLM or cause it to start hallucinating information. Furthermore, it becomes much more expensive as it 
# has to process and send more information.

print()
print("-="*20 + "-")
print()

# LlamaIndex Q3
query = "What does BrightLeaf do, who have they worked with, and what are their benefits? Also who is their CEO?"
query_engine = index.as_query_engine(similarity_top_k=5)

print("Question:", query)
response = query_engine.query(query)
print("Answer:", response)
for this_node in response.source_nodes:
    print("Node ID:", this_node.node.node_id)
    print(f"Similarity Score: {this_node.score:.4f}")
    print("First 150 Chars:", this_node.node.get_content()[:150])
    print()

"""
Expected: Retrieve multiple articles and mix information to make it seem muddied together.

Actually: Broke down the information into each section, retrieved the relevant information, and properly relayed it.
The model also responded by saying "The information provided does not specify who the CEO of BrightLeaf is."

Change: Adjusting the similarity_top_k to include more/less information to improve the scope of the answer.
"""


# LlamaIndex Q4

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

query_engine = index.as_query_engine(similarity_top_k=3)
faithful_eval = FaithfulnessEvaluator(llm=llm)
relevance_eval = RelevancyEvaluator(llm=llm)

# 1st run
query = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(query)

faithfulness_result = faithful_eval.evaluate_response(query=query, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevance_eval.evaluate_response(query=query, response=response)
print("Relevancy Result: " + str(relevancy_result.score))
print()
# 2st run
query = "Who created and runs the BrightLeaf company?"
response = query_engine.query(query)

faithfulness_result = faithful_eval.evaluate_response(query=query, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevance_eval.evaluate_response(query=query, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

"""
1.  A faithfulness score of 1.0 would represent YES in the context of is the response faithful to the original
    text? A score of 0.0 would represent NO in the same context and means it is hallucinated.

2.  A relevancy score of 1.0 would represent YES in the context of is the response staying on topic? A score of
    0.0 would represent NO in the same context.

3.  Yes, the relevancy score changed from 1.0 in the 1st run to 0.0 in the second run. That is because there is
    no information relating to who created the company so it has to explain that it doesn't know or doesn't have 
    the information to answer the question.

4.  The "LLLM-as-a-judge" aproach is used because there is no correct answer, and so it can't use an exact match
    to grade the answer.
"""