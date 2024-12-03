from flask import Flask, request, jsonify
from services.waha import Waha
import time
import random
from bot.ai_bot import AIBot

app = Flask(__name__)

@app.route('/chatbot/webhook/',methods = ['POST'])
def webhook():
    data = request.json

    print(f'EVENTO RECEBIDO: {data}')

    waha = Waha()
    ai_bot = AIBot()

    chat_id = data['payload']['from']
    received_message = data['payload']['body']

    isgroup = '@g.us' in chat_id
    isstatus = 'status@broadcast' in chat_id

    if isgroup or isstatus:
        return jsonify({'status':'success', 'message':'Mensagem de grupo/status ignorada.'})

    waha.start_typing(chat_id=chat_id)
    time.sleep(random.randint(2,4))
    history_messages = waha.get_history_messages(
        chat_id=chat_id,
        limit=10
    )

    response = ai_bot.invoke(
        history_messages=history_messages,
        question=received_message)

    waha.send_message(
        chat_id = chat_id,
        message = response,
    )

    waha.stop_typing(chat_id=chat_id)

    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
