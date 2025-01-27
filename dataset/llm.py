import openai
import pandas as pd 



def caption(contents,prompt):

    conversation=[]
    conversation.append({"role": "system", "content": contents})
    conversation.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=conversation
    )
    answer = response.choices[0].message.content.strip()
    

    return answer


if __name__ == "__main__":

    

