from dotenv import load_dotenv
from openai import OpenAI
import json



# -=-=- Task 1 -=-=-
load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

system_prompt = """
You are a professional Job Application Coach who helps people changing careers or people trying to get a job in general.
You will help these users improve their resumes, generate cover letters, and other prep needed for job applications and interviews.
Follow these rules:
1. Stay focused on job application materials.
2. Always remind the user to review and edit its output before submitting anywhere.
3. Acknowledge that it may not know the user's specific industry norms, and that the user should use their own judgement.
4. Be supportive and encouraging whilst staying realistic to the user.
5. Use clear and concise professional language.
6. Never add any degrees, qualifications, or certifications. instead, ask the user if they have any and add the ones they do only if mentioned.
7. Do not add Markdown text or backtick fences. Do not use asterisks to make bold or italicized text. You can use numbers to make a list.
"""
# I added a rule to make sure to never add any documentable qualifications unless they are 





# -=-=- Task 2 -=-=-
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original. Do not invent
    numbers unless they are specifically provided.

    Here is an example:
        original: Made reports for the management team.
        improved: Compiled and presented comprehensive reports to the management team, facilitating informed decision-making.

    Return ONLY a valid JSON list and no other text. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).
    Do not include any markdown or backticks on the JSON.

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    raw_response = get_completion(messages)
    
    try:
        loaded_json = json.loads(raw_response)
    
    except json.JSONDecodeError:
        print("Error: Response was not valid JSON")
        print("Raw Response:", raw_response)

    return loaded_json
    

# These bullet points are weak because they are too generic. They don't explain how each
# piece did what it did, or what exactly the improvement does and why it's a benefit. The
# model added information explaining what actions the user takes to benefit the customer,
# and what benefits it adds to the company
# ex. "Helped customers with their problems" -> "Resolved customer issues by provided tailored solutions, enhancing satisfaction and loyalty"
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

"""
new_bullets = rewrite_bullets(bullets)

for dict_item in new_bullets:
    for key in dict_item:
        print(key, "->", dict_item[key])
print()
"""





# -=-=- Task 3 -=-=-
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.
    Respond only with the opening's text.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]

    return get_completion(messages)
"""
job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Pandas."

print(generate_cover_letter(job_title, background))
"""
# These examples were chosen to help the chatbot understand the general format it should follow. The
# Few shot pattern helps explain how long the person has been working at a given company, along with
# what they did while working there so they can transition into applying those previous responsibilities
# into their future roles





# -=-=- Task 4 -=-=-
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if not flagged:
        return True

    print("Message was flagged. Please revise it.")
    #print(result.results[0].categories.model_dump())
    return False
"""
print(is_safe("I want to kill my neighbor.")) -> prints returns False and prints the categories with issues (harassment and harassment_threatening)
print(is_safe("I want to water my neighbor's plant.")) -> returns True
"""





# -=-=- Task 5 -=-=-
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            print()
            print(f"Job Application Helper: Here are the newly rewritten bullet points:\n")
            new_bullets = rewrite_bullets(raw_bullets)
            for dict_item in new_bullets:
                for key in dict_item:
                    print(key, "->", dict_item[key])
                print()

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            print(f"Job Application Helper: Here is the newly generated cover letter opening:\n")
            print(generate_cover_letter(job_title, background))
            print()

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})

            # - Call get_completion(messages)
            response = get_completion(messages)

            # - Print the reply
            print("Job Application Helper:\n", response)

            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": response})
            #print(len(messages))
            
            print()


if __name__ == "__main__":
    run_chatbot()





# -=-=- Task 6 -=-=-
# Option A:
"""
1. The bot is trained on specific text and given examples of how it should respond, so it will use
similar language and styles to rewrite or generate any content based on that. This means that if
it speaks in a diction that is completely unnatural to you, it might be obvious that you used AI
to write some of these things. What's even worse is if you use the information from the AI without
understanding what it is actually saying ro means in it's interpretation. 

2. There could be some wrong information or fabricated resulkts such as wrong levels of experience
and qualifications or it could include placeholder text like "[Company Name]" which would show you
are using AI to generate the content without double checking anything you're doing and then also
lying to the reviewer on your resume/application.

3. A disclaimer is the bare minimum to add, as it is such a simple-level tool that there can and will
be mistakes that are unaccounted for. As a result, each user should properly double check anything
that they recieve from the AI to ensure that it is properly working. I would also add better working
rules and prompts for the AI to follow. Without complete testing (such as using a large population
sample), it is hard to test how the AI should handle and approach different scenarios.

"""