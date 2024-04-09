from flask import Flask, request, jsonify, render_template
from main import main_fight,predict
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    video = request.files['video']
    video_path = "Videos/" + video.filename
    video.save(video_path)
    result_main_fight = main_fight(video_path)
    result_predict = predict(video_path)

    main_fight_data = json.loads(result_main_fight)
    predict_data = json.loads(result_predict)

    return render_template('results.html', main_fight_data=main_fight_data, predict_data=predict_data)

if __name__ == '__main__':
    app.run(debug=True)
