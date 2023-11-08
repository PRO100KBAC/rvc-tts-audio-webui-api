from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)
gradio_client = Client("http://127.0.0.1:7860/")

@app.route('/rvctts', methods=['POST'])
def tts_process():
    try:
        input_text = request.json['text']
        input_speaker = request.json['model']
        input_volume = request.json['volume']
        input_gain = request.json['gain']
        input_edgetts_model = request.json['edgetts_model']
        input_edgetts_speed = request.json['edgetts_speed']
        input_transpose = request.json['transpose']
        input_indexrate = request.json['indexrate']
        input_protect = request.json['protect']
        input_result = request.json['res']

        result = gradio_client.predict(
            input_speaker,
            input_transpose,
            input_volume,
            input_gain,
            True,
            input_indexrate,
            input_protect,
            input_text,
            input_edgetts_model,
            input_edgetts_speed,
            0,
            0,
            0,
            "mp3",
            128,
            api_name="/predict"
        )

        audio_path = result[input_result]

        return (audio_path)
    except Exception as e:
        return (str(e)), 500

@app.route('/rvcaudio', methods=['POST'])
def audio_file_process():
    try:
        input_speaker = request.json['model']
        input_volume = request.json['volume']
        input_gain_vc = request.json['gain_vc']
        input_gain_mus = request.json['gain_mus']
        input_url = request.json['url']
        input_slow = request.json['slow']
        input_transpose = request.json['transpose']
        input_indexrate = request.json['indexrate']
        input_protect = request.json['protect']
        input_result = request.json['res']
        input_ff_multi = request.json['multi']
        input_ff_duration = request.json['duration']

        result = gradio_client.predict(
            input_speaker,
            input_transpose,
            input_volume,
            input_gain_mus,
            input_gain_vc,
            True,
            input_indexrate,
            input_protect,
            "youtube-url",
            input_url,
            "empty.mp3",
            input_slow,
            True,
            True,
            input_ff_multi,
            input_ff_duration,
            0,
            0,
            0,
            "mp3",
            128,
            api_name="/predict_1"
        )

        audio_path = result[input_result]

        return (audio_path)
    except Exception as e:
        return (str(e)), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7850)
