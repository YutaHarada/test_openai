from langchain.utilities import BingSearchAPIWrapper
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableParallel, RunnableLambda
from typing import  Any, Dict
from operator import itemgetter


# ツールの準備
def format_result(query: str) -> Dict:
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

# システムプロンプトの準備
system = "You are a helpful assistant."
system_message_prompt = SystemMessagePromptTemplate.from_template(system)

# OutputParserの準備
json_parser = SimpleJsonOutputParser()

# Prompt①の準備
template_1 = """questionを検索エンジンの検索クエリに変換し'query'キーの値としてください。

 {question}
"""

human_prompt_1 = PromptTemplate(
    template=template_1,
    input_variables=["question"],
)
human_message_prompt_1 = HumanMessagePromptTemplate(prompt=human_prompt_1)

chat_prompt_1 = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt_1]
)

# LLMモデル①の初期化
model_1 = ChatOpenAI(temperature=0)

# =================================================================

# Prompt②の準備
template_2 = """snippetの内容に基づいてquestionに対して丁寧に回答してください。

{snippet}

{question}
"""

human_prompt_2 = PromptTemplate(
    template=template_2,
    input_variables=["snippet", "question"],
)
human_message_prompt_2 = HumanMessagePromptTemplate(prompt=human_prompt_2)

chat_prompt_2 = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt_2]
)

# LLMモデル②の初期化
model_2 = ChatOpenAI(temperature=0.8)


class CustomCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        self.llm_result : LLMResult
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.llm_result = response


def extract_snippet(x: Dict):
    return x['snippet']

def extract_links(x: Dict):
    return x['links']

qa_chain = (
RunnableParallel({
    "context": chat_prompt_1 | model_1 | json_parser | search ,
    "question": itemgetter("question")
})
| RunnableParallel({
    "snippet": itemgetter("context") | RunnableLambda(extract_snippet),
    "links": itemgetter("context") | RunnableLambda(extract_links),
    "question": itemgetter("question")
})
| RunnableParallel({
    "response": chat_prompt_2 | model_2,
    "links": itemgetter("links")
})
)

callback = CustomCallbackHandler()
question = "2023年の紅白歌合戦は何組が勝ちましたか？"
qa_chain.invoke({"question": question}, config={"callbacks": [callback]})
