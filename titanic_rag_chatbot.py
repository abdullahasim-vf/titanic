import os
import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

def load_data(csv_path="Titanic-Dataset.csv"):
    return pd.read_csv(csv_path)

def row_to_text(row):
    age = f"{row['Age']} years old" if pd.notna(row['Age']) else "age unknown"
    fare = f"${row['Fare']:.2f}" if pd.notna(row['Fare']) else "fare unknown"
    return (f"Passenger {row['PassengerId']}: {row['Name']}, {row['Sex']}, {age}, "
            f"{row['Pclass']} class, fare {fare}, "
            f"{'survived' if row['Survived'] == 1 else 'did not survive'}.")

def prepare_full_context(df):
    df['text_chunk'] = df.apply(row_to_text, axis=1)
    return "\n".join(df['text_chunk'].tolist())

def ask_llm(question, full_context):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        max_output_tokens=900, 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    template = """
    You are a Titanic dataset expert. Use the following data to answer the question.
    If the question is not related to the Titanic dataset, politely explain that.

    Data:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"context": full_context, "question": question}).strip()

df = load_data()
context = prepare_full_context(df)
response = ask_llm("How many 3rd class passengers survived?", context)
print(response)
