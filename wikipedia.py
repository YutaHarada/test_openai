from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import  WikipediaRetriever

from langchain_core.runnables import RunnableParallel
from operator import itemgetter

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage # NOQA

callback = CustomCallbackHandler()

llm = ChatOpenAI()
retriever = WikipediaRetriever(
    lang="ja",
    top_k_result=3,
    doc_content_chars_max=500,
)

human_prompt = ChatPromptTemplate.from_template(
    """以下のcontextだけに基づいて回答してください。

{context}

質問: {question}
"""
)

system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
human_message_prompt = HumanMessagePromptTemplate.from_template("""以下のcontextだけに基づいて回答してください。

{context}

質問: {question}
"""
)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

qa_chain = (
{
    "context": itemgetter("question") | retriever,
    "question": itemgetter("question")
}
| RunnableParallel({
    "response": chat_prompt | llm,
    "context": itemgetter("context"),
})
)
