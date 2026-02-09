
import time
import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import google.genai as genai
def create_embeddings(text):
    r=requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input":text
    })
    embedding=r.json()['embeddings'][0]
    return embedding

def incomming_query(question:str)-> str:
    question_embedding=create_embeddings(question)
    

# incomming_query=input("\nAsk a question:")
# question_embedding=create_embeddings(incomming_query)
#print(question_embedding)


#find similarities of question_embeddings with all embeddings with cosin simalarities:
# load stored embeddings
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "text_to_embedding.json")
    df = pd.read_json(DATA_PATH)

    # cosine_similarity expects 2D arrays
    query_vector = [question_embedding]

    document_vectors = df["embedding"].tolist()

    # compute cosine similarity
    similarities = cosine_similarity(query_vector, document_vectors)

    # similarities is 2D → extract first row
    df["similarity"] = similarities[0]

    # sort by similarity (highest first)
    df_sorted = df.sort_values(by="similarity", ascending=False)

    # show top 3 most relevant chunks
    top_5=(df_sorted[["start", "end", "text", "similarity"]].head(5))
    # print(top_5)


    #now i want this data frame in text so the code for this is below 
    context = ""
    for _, row in top_5.iterrows():
        context += f"""
    [start: {row['start']} - end: {row['end']}]
    {row['text']}
    """
    #print(context)




    #LLM from here maybe this would be the final file ...gettign respoose form llm
    prompt=f'''{question}
    In this video the guy is teach a Exploratory Data Analysis project. Here are the video start time in seconds
    end time in seconds and the text at that time: {context}

    -------------------------
    {question}
    -> User asked this question related to the video, you have to answer the question and how much context is taught.
    (only tell time stapms when user ask about time and where it is taught other wise dont mention time)
    -> If user ask unrealated question,tell him that you can only ask question realted to the video/course and gide him.
    -> Convert that time stamp seconds into minutes for better understadning of the user.
    -> If use say hi or hello or any other greeting, then reply user with, "hello how may i help you..you can ask me any 
        question regding to this course" or sometimes tweak this same sentence and ask differently.
    -> Remember dont act suspecious like you have been provided any data .. you have to act like you are a LLM or a human.
    '''




    client = genai.Client(api_key="hh")

    # response = client.models.generate_content(
    #     model="models/gemini-2.5-flash",
    #     contents=prompt
    # )

    # print(response.text)

    stream = client.models.generate_content_stream(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    # for chunk in stream:
    #     if chunk.text:
    #         print(chunk.text, end="", flush=True)
    full_answer = ""

    for chunk in stream:
        if chunk.text:
            full_answer += chunk.text

    return full_answer


if __name__ == "__main__":
    q = input("\nAsk a question: ")
    answer = incomming_query(q)
    print(answer)