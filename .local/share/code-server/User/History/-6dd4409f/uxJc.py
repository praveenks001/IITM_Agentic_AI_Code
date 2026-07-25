"""hello_llm.py — Week 1 practice: our first call to an LLM.

Run from the repo root:
    python practice_examples/week01/hello_llm.py "What is RAG in one sentence?"

If you dont pass any question, it will give the default answer

To run:
python hello_llm.py "What is Agentic Ai and how it helps in trading?" > docs/runs_output/01_output_as_is.txt
python pythonfilepath question > outputfolder with file name
"""



import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()          # loads OPENAI_API_KEY from .env on your own machine
client = OpenAI()      # finds the key in the environment automatically


def ask(question: str) -> str:
    """Send one question to the model, return the answer text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini", # define the llm model could be anything
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": question},
        ],
        temperature=0., #temperature in LLM related to creativity of the answer returned by LLM whose input values ranging from 0-2 where 0-0.3 is less creative , 0.0
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Say hello in one sentence."
    print(ask(q))

