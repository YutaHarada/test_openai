# langchain == 0.0.351
# langchain_core == 0.1.8

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import  WikipediaRetriever

from langchain_core.runnables import RunnableConfig, RunnableParallel, RunnablePassthrough
from operator import itemgetter

callback = CustomCallbackHandler()

llm = ChatOpenAI()
retriever = WikipediaRetriever(
    lang="ja",
    top_k_result=3,
    doc_content_chars_max=500,
)

prompt = ChatPromptTemplate.from_template(
    """以下のcontextだけに基づいて回答してください。

{context}

質問: {question}
"""
)

qa_chain = (
{
    "context": itemgetter("question") | retriever,
    "question": itemgetter("question")
}
| RunnableParallel({
    "response": prompt | llm,
    "context": itemgetter("context"),
})
)

result = qa_chain.invoke({"question":"姫路城について教えてください"}, config={'callbacks': [callback]})
