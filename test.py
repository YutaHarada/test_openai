from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 回答をストリーミング形式で取得
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! I'm John."}
    ],
    stream=True)

for chunk in response:
    choice = chunk.choices[0]
    if choice.finish_reason is None:
        print(choice.delta.content)


# ========================================================================

# from openai import AzureOpenAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# client = AzureOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_version="2023-12-01-preview"
# )

# # 回答をストリーミング形式で取得
# response = client.chat.completions.create(model="gpt-3.5-turbo",
#                                           messages=[
#                                               {"role": "system", "content": "You are a helpful assistant."}, # NOQA
#                                               {"role": "user", "content": "Hello! I'm John."}  # NOQA
#                                               ],
#                                           stream=True)

# for chunk in response:
#     print(chunk)
#     # choice = chunk.choices[0]
#     # if choice.finish_reason is None:
#     #     print(choice.delta.content)
