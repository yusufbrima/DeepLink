from flask import Flask,render_template
from pathlib import Path
import numpy as np
from flask import request
from flask import jsonify
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import librosa
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pylab as plt

# import pylab as plt
import librosa.display
app = Flask(__name__)


@app.route('/plot/<p_id>')
def plot_png(p_id):
    # message = request.get_json(force=True)
    if(int(p_id) == 1):
        fig = create_figure(1)
    else:
        fig = create_figure(2)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(n):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    audios =  ['choice', 'vibeace','fishin','humpback','nutcracker']
    idx =  np.random.randint(0, len(audios))
    y,sr = librosa.load(librosa.ex(audios[idx]), sr=None,mono=True)
    if(n  ==1):
        librosa.display.waveplot(y=y, sr=sr, ax=axis)
    else:
        spec =  librosa.stft(y)
        n_spec = librosa.amplitude_to_db(np.abs(spec))
        librosa.display.specshow(n_spec, sr=sr, ax=axis, x_axis='time', y_axis='log')
    return fig

@app.route("/audio", methods=["POST",'GET'])
def audio():
    if request.method == 'POST':
        return "Hello world coming soon :)"
    else:
        fname =  librosa.ex('vibeace')
        y,sr = librosa.load(fname, sr=None,mono=True)
        return render_template('audio.html', data = {'fname': '/plot/1','fname2': '/plot/2'})


@app.route("/test", methods=["POST",'GET'])
def test():
    if request.method == 'POST':

        message = request.get_json(force=True)
        x =  np.array([message['x1'],message['x2'] ,message['x3'] ,message['x4']  ], dtype=np.float).reshape(1,4)
        scaler =  StandardScaler()
        X =  scaler.fit_transform(x)
        model = tf.keras.models.load_model('./models/DeepSpeaker.h5')
        pred =  model.predict(X)
        yhat =  np.argmax(pred, axis=1)
        return jsonify({'greeting':   pred.reshape(4,).tolist() })
    else:
        return render_template('test.html')




@app.route("/hello", methods=["POST"])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {'greeting': f'Hello, {name}!' }
    return jsonify(response) 

@app.route('/inference', methods=['POST','GET'])
def inference():
    return render_template('model.html')


@app.route('/', methods=['GET', 'POST'])
def home():
   return render_template('index.html')

if __name__ == "__main__":
    
    app.run(debug=True)