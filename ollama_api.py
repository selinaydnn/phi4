from fastapi import FastAPI
from pydantic import BaseModel
from ollama import chat, ChatResponse
from warnings import filterwarnings


filterwarnings("ignore")


app = FastAPI()



def chat_reponse(input_message):
    user_message_info = """
    \nOgrencinin sordugu soru: {input_message} \n
    """
    user_message_info += 'benim bu soru ile ilgili buldugum chunk listesi {mesaj}'

    response: ChatResponse = chat(model='phi4', messages=[
        {
            'role': 'user',
            'content': f"{user_message_info}",
        },
    ])

    content = response['message']['content']
    return content


@app.get("/chat-response/")
def get_chat_response(user_input):
    try:
        result = chat_reponse(user_input)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}