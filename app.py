# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
import openai  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv
import asyncio

from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI as ac
import doug
import rqa


load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

documents = doug.dg_read_file("data/KingLear.txt")
split_documents = doug.dg_split_file(documents)

vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))

chat_openai = ac()
retrieval_augmented_qa_pipeline = rqa.RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai
)

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)




    number_of_chunks = len(split_documents)

    content = f"Doug is saying something. The number of chunks is {number_of_chunks}"
    await cl.Message(
        content=content,
    ).send()


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: str):
    settings = cl.user_session.get("settings")

    """prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message),
            ),
        ],
        inputs={"input": message},
        settings=settings,
    )
    

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await openai.ChatCompletion.acreate(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0]["delta"].get("content", "")
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt
    """

    response = retrieval_augmented_qa_pipeline.run_pipeline(message)
    msg = cl.Message(content=response)

    # Send and close the message stream
    await msg.send()
