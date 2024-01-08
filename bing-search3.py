bfrom dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser

from langchain.utilities import BingSearchAPIWrapper
from langchain.tools import Tool

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage # NOQA

from langchain.callbacks.base import BaseCallbackHandler
from typing import  Any, Dict
from langchain_core.runnables import RunnableConfig
from langchain_core.outputs import LLMResult


class CustomCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        self.llm_result : LLMResult
        self.chain_result : Dict
        self.tool_result : str
        self.agent_result : AgentFinish
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        # self.llm_result = response.generations[0][0].generation_info
        self.llm_result = response
    
    # def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
    #     self.chain_result = outputs
    
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        self.tool_result = output

    # def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
    #     self.agent_result = finish

callback_1 = CustomCallbackHandler()
callback_2 = CustomCallbackHandler()
        
load_dotenv()

# 質問文
question = "2023年の紅白歌合戦は何組が勝ちましたか？"


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


description = """与えられた検索クエリと検索エンジンBingを用いて調査を行ってください。
"""

search = Tool(
    name="CustomBingSearch", description=description, func=format_result
)


# OutputParser①の準備
json_parser = SimpleJsonOutputParser()


# Prompt①の準備
template_1 = """questionを検索エンジンの検索クエリに変換し'query'キーの値としてください。

 {question}
"""

human_prompt_1 = PromptTemplate(
    template=template_1,
    input_variables=["question"],
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

chain_1 = chat_prompt_1 | model_1 | json_parser | search

search_result = chain_1.invoke({"question": question},config={'callbacks': [callback_1]})

# =================================================================

# Prompt②の準備
template_2 = """snippetの内容に基づいてquestionに対する丁寧な回答を作成してください。

{snippet}

{question}
"""

human_prompt_2 = PromptTemplate(
    template=template_2,
    input_variables=["snippet", "question"],
)

system_message_prompt_2 = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
human_message_prompt_2 = HumanMessagePromptTemplate(prompt=human_prompt_2)

chat_prompt_2 = ChatPromptTemplate.from_messages(
    [system_message_prompt_2, human_message_prompt_2]
)

# LLMモデルの初期化
model_2 = ChatOpenAI(temperature=0)

# Chainの準備
chain_2 = chat_prompt_2 | model_2

# 実行
result = chain_2.invoke(
    {"snippet": search_result['snippet'],"question": question},
    config={'callbacks': [callback_2]}
)

# for i in result:
#     print(i.content)

result.content
