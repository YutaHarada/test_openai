from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from langchain.utilities import BingSearchAPIWrapper
from langchain.tools import Tool

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage # NOQA


load_dotenv()

# 質問文
question = "大谷翔平の現在の年俸は？"


# ツールの準備
def format_result(query: str) -> dict:
    """ Bing検索結果のスニペットの統合とリンクリストを生成する関数

    Paramater
    --------
    quey: 検索に使用するクエリ
    n: 検索するドキュメントの数

    Return
    ------
    dict: Bing検索結果のスニペットの統合とリンクリスト

    """
    bing = BingSearchAPIWrapper()
    results = bing.results(query, 4)
    snippets = []
    links = []
    for result in results:
        snippets.append(result["snippet"])
        links.append(result["link"])
    snippet = " ".join(snippets)
    return {"snippet": snippet, "links": links}


description = """与えられたqueryについて検索エンジンBingを用いて調査を行ってください。
検索結果として得られた記事の内容は結合し、参考にしたWebサイトのURLを記録してください。
"""

search = Tool(
    name="CustomBingSearch", description=description, func=format_result
)


# OutputParser①の準備
response_schemas_1 = [
    ResponseSchema(name="query", description="ユーザーのquestionに対する検索エンジンの検索クエリ"),
]
output_parser_1 = StructuredOutputParser.from_response_schemas(response_schemas_1)  # NOQA

# Prompt①の準備
format_instructions_1 = output_parser_1.get_format_instructions()

template_1 = """ユーザーのquestionを検索エンジンの検索クエリに変換してください。

{format_instructions}

{question}

"""

human_prompt_1 = PromptTemplate(
    template=template_1,
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions_1},
)

system_message_prompt_1 = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant."
    )
human_message_prompt_1 = HumanMessagePromptTemplate(prompt=human_prompt_1)

chat_prompt_1 = ChatPromptTemplate.from_messages(
    [system_message_prompt_1, human_message_prompt_1]
)

# LLMモデルの初期化
model_1 = ChatOpenAI(temperature=0)

chain_1 = chat_prompt_1 | model_1 | output_parser_1 | search

search_result = chain_1.invoke({"question": question})
print(search_result)

# =================================================================

# OutputParser②の準備
response_schemas_2 = [
    ResponseSchema(name="result", description="与えられた'snippet'の値を要約した文章"),
]
output_parser_2 = StructuredOutputParser.from_response_schemas(
    response_schemas_2
)

# Prompt②の準備
format_instructions_2 = output_parser_2.get_format_instructions()

template_2 = """与えられた入力に対して以下の処理を施した上で辞書型で出力してください。
・"snippet"の内容を用いて"question"に対する回答を作成し、キー"result"の値として格納してください。

{format_instructions}

{snippet}

{question}
"""

human_prompt_2 = PromptTemplate(
    template=template_2,
    input_variables=["snippet", "question"],
    partial_variables={"format_instructions": format_instructions_2},
)

system_message_prompt_2 = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant."
    )
human_message_prompt_2 = HumanMessagePromptTemplate(prompt=human_prompt_2)

chat_prompt_2 = ChatPromptTemplate.from_messages(
    [system_message_prompt_2, human_message_prompt_2]
)

# LLMモデルの初期化
model_2 = ChatOpenAI(temperature=0)

# Chainの準備
chain_2 = chat_prompt_2 | model_2 | output_parser_2

# 実行
result = chain_2.invoke(
    {
        "snippet": search_result['snippet'],
        "question": question
    })

print(result)
