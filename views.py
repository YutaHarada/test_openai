# import文
import os
import queue
import json

from flask import (
    Blueprint,
    request,
    render_template,
    Response
)

from langchain.chat_models import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.retrievers import WikipediaRetriever
from langchain.callbacks.base import BaseCallbackHandler


class CustomCallbackHandler(BaseCallbackHandler):

    def __init__(self, que):
        self.que = que

    # LLMからの応答ストリームによって文言を受け取るたびに呼び出されるメソッド
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.que.put(token)


# chatアプリの生成
chat = Blueprint(
    name="chat",
    import_name=__name__,
    template_folder="templates"
)

# ストーリミング形式の応答を順番に格納するためのキューの準備
que = queue.Queue()


# ルートエンドポイント
@chat.route('/')
def index():
    return render_template('chat/index.html')


# 回答作成用エンドポイント
@chat.route('/ask', methods=['POST'])
def ask():

    # Language modelsの初期化(質問生成用)
    llm = ChatOpenAI(
        model=os.environ["OPENAI_API_MODEL"],
        temperature=os.environ["OPENAI_API_TEMPERATURE"],
        streaming=True,
        callbacks=[CustomCallbackHandler(que)]
    )

    # Retrieverの設定
    retriever = WikipediaRetriever(
        lang="ja",
        top_k_result=3,
        doc_content_chars_max=1000,
    )

    # Chainの設定
    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    # リクエストのJSONボディからデータを取得
    question = request.get_json("question")

    # 回答の取得
    answer = qa_chain(question["question"])

    url_list = []
    for content in answer["source_documents"]:
        url_list.append(content.metadata['source'])

    return Response(json.dumps({"urlList": url_list}, ensure_ascii=True))


# SSE用エンドポイント
@chat.route('/listen')
def listen():
    def stream():
        while True:
            msg = que.get()
            if msg is None:
                break
            yield f'data: {msg}\n\n'
    response = Response(stream(), mimetype='text/event-stream')
    return response
