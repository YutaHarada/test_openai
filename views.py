# import文
import os
from dotenv import load_dotenv
from openai import OpenAI
from flask import (
    Blueprint,
    render_template,
    Response,
)

load_dotenv()

# chatアプリの生成
chat = Blueprint(
    name="chat",
    import_name=__name__,
    template_folder="templates"
)


# ルートエンドポイント
@chat.route('/')
def index():
    return render_template('chat/index.html')


# 回答作成用エンドポイント
@chat.route('/listen')
def ask():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 回答をストリーミング形式で取得
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! I'm John."}
        ],
        stream=True)

    def stream():
        for chunk in response:
            choice = chunk.choices[0]
            if choice.finish_reason == "stop":
                break
            yield f'data: {choice.delta.content}\n\n'

    return Response(stream(), mimetype='text/event-stream')
