from dotenv import load_dotenv
from openai import OpenAI


# -=-=- The Chat Completions API -=-=-
# -=- API Question 1 -=-
print("-=- API Question 1 -=-")
load_dotenv()
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print(f"Content Output: {response.choices[0].message.content}")
print(f"Model Type: {response.model}")
print(f"Token Amount: {response.usage.total_tokens}")
print()



# -=- API Question 2 -=-
print("-=- API Question 2 -=-")
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]

for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )

    print("Temperature:", temp)
    print("Response:", response.choices[0].message.content)
    print()
    # The outputs slightly differ with the first 2, but the 3rd is significantly different. The first 2 have the same names,
    # except the 0 response has a keyword after the company name to narrow its title down more. The 0.7 response omits that
    # keyword and just gives simpler less-specified name. The final tempertaure of 1.5 gives a single response for a name.
    #
    # To keep a consistent or reproducible output, you would want to use temperature = 0.0 as it always gives the top result item.



# -=- API Question 3 -=-
print("-=- API Question 3 -=-")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

for item in response.choices:
    print("Result:", item.message.content)
print()



# -=- API Question 4 -=-
print("-=- API Question 4 -=-")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)
print(f"Content Output: {response.choices[0].message.content}")
print()
# Setting max_tokens to 15 cutoff the message response mid sentence. This is useful when you want to limit the cost
# of token usage or if you have a max word limit that must not be crossed for your response.





# -=-=- System Messages and Personas -=-=-
# -=- System Question 1 -=-
print("-=- System Question 1 -=-")
print("Persona 1:")
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(response.choices[0].message.content)
print()

print("Persona 2:")
messages = [
    {"role": "system", "content": "You are an angry, irritated Python tutor. You are fed up and explain things quickly and rudely."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(response.choices[0].message.content)
print()
# The personality of the model changed as we gave it a role that has a different context as the other one. 



# -=- System Question 2 -=-
print("-=- System Question 2 -=-")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print(response.choices[0].message.content)
print()
# The model knows Jordan's name because we passed it as an argument into this one single call. The whole conversation is sent
# in the single API call so it's not remembering the previous message, it is all one whole message.





# -=-=- Prompt Engineering -=-=-
# -=- Prompt Question 1 -=-
print("-=- Prompt Question 1 -=-")
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Classify the following as positive, negative, or mixed: {review}"}],
    )
    print(f"Review #{i+1}: {response.choices[0].message.content}")
    print()



# -=- Prompt Question 2 -=-
print("-=- Prompt Question 2 -=-")
for i, review in enumerate(reviews):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""Classify the following as positive, negative, or mixed. 
        Example: Review: "Fast shipping but the item arrived damaged."Sentiment: mixed. 
        
        {review}"""}],
    )
    print(f"Review #{i+1}: {response.choices[0].message.content}")
    print()
    # The output changed from the first set to follow the example provided. It responded with 2 words, sentiment and it's value of 
    # positive, negative, or mixed



# -=- Prompt Question 3 -=-
print("-=- Prompt Question 3 -=-")
for i, review in enumerate(reviews):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""Classify the following as positive, negative, or mixed. 
        Example 1: Review: "Fast shipping but the item arrived damaged."Sentiment: mixed. 
        Example 2: Review: "No response from company and item was not recieved."Sentiment: negative. 
        Example 3: Review: "Great quality and easy process."Sentiment: positive.
        
        {review}"""}],
    )
    print(f"Review #{i+1}: {response.choices[0].message.content}")
    print()
    # Each approach has its benefits
    # Zero-Shot: Quick and easy; Gets your result and doesn't have to jump through hurdles to return it in a specific format
    #   I would choose this for a simple query when asking for something without requiring a structure.
    # One-Shot: Happy medium; Simple and useful for provided a base format that should be followed. However can be hard to categorize exactly on large scales of options
    #   I would choose this for a project such as the one above. It only needs one example to understand the formula.
    # Few-Shot: Expensive but exact; Provides the best accuracy and formatting for the results but requires more tokens to explain that formatting.
    #   I would choose this for a much more expansive version above, where there are plenty of options and some can be similar to each other.



# -=- Prompt Question 4 -=-
print("-=- Prompt Question 4 -=-")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"""Solve the following problem:
    "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
    takes a new job that pays $7,500 more per year than her post-raise salary.
    What is her final annual salary?"

    Show your reasoning step by step before giving a final answer. Label the final answer clearly.
    """}],
)

print("Response:", response.choices[0].message.content)
print()
# The step by step improves accuracy because it breaks down the whole larger problem into smaller pieces
# than can be looked at and checked individually.



# -=- Prompt Question 5 -=-
print("-=- Prompt Question 5 -=-")
import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"""Analyze the review below:
    {review}

    Response only as valid JSON with the keys as sentiment, confidence (a float from 0 to 1), and reason (one sentence).
    Do not invlude any markdown or backticks.
    """}],
)
print("Response:", response.choices[0].message.content)

try:
    loaded_json = json.loads(response.choices[0].message.content)
    print("Sentiment:", loaded_json["sentiment"])
    print("Confidence:", loaded_json["confidence"])
    print("Reason:", loaded_json["reason"])
    
except json.JSONDecodeError:
    print("Error: Response was not valid JSON")
    print("Raw Response:", response.choices[0].message.content)
print()



# -=- Prompt Question 6 -=-
print("-=- Prompt Question 6 -=-")
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print("Response:\n", response.choices[0].message.content)
print()

user_text = "Today I went running an saw a friend when I passed by the park. We talked \
    for a bit about how we love dogs. He said his favorite dog is a poodle!"
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
)
print("Response:\n", response.choices[0].message.content)
print()
# Delimiters help prevent any confusion about what text is supposed to be analyzed be the model.
# Sometimes, it might think the text is still part of the instructions so it continues to wait
# for the text thinking you haven't yet provided it. Other times, it might do the opposite
# and analyze the instructions you sent as the text.





# -=-=- Local Models with Ollama -=-=-
# -=- Ollama Question 1 -=-
print("-=- Ollama Question 1 -=-")
"""
Thinking...
Okay, the user wants me to explain a large language model in two sentences. Let me start by recalling the basics. A large language model is a type of artificial intelligence that can understand and 
generate human-like text. I should mention its capabilities, like understanding context and generating complex sentences.

Wait, the user specified two sentences, so I need to make sure each sentence is concise. First sentence: Introduce the model as an AI capable of understanding and generating human-like text. Second 
sentence: Highlight its ability to process large amounts of data and generate context-aware responses. Check for clarity and correctness. Maybe add examples if needed, but since it's two sentences, 
keep it simple. Make sure it's in two separate sentences without any markdown. Alright, that should cover it.
...done thinking.

A large language model is an AI system that can understand and generate human-like text, learning from vast amounts of data to understand context and produce meaningful, context-aware responses.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what a large language model is in two sentences."}],
)
print("Response:", response.choices[0].message.content)
print()

# The first difference is that the Ollama gives the full thinking process along with the result, but also
# the result from Ollama is in one sentence despite me asking for two sentences. The OpenAI response also
# seemed to be more refined and dignified when compared to the Ollama
#
# One advantage for running a model locally is that it is significantly cheaper and private as it relies 
# on your own hardware instead of spending tokens in a datacenter. One disadvantage though, is that it
# requires more modern hardware to run efficiently and quickly and it is much simpler so it produces 
# simpler results.
