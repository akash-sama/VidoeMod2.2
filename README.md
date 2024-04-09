_NOTE: badcode contains different methods and building model and should be ingnored_



Before running the application, you need to install the necessary packages listed in requirements.txt. Run the following command:

### Install Required Packages

Before running the application, you need to install the necessary packages listed in `requirements.txt`. Run the following command:

```bash
pip install -r requirements.txt
```

then

```bash
python app.py
```

The program should run on the local servear and the website can be viewed on

 http://127.0.0.1:5000/

The order of the files is importent and will not work if changed

### General Overview of the Modles:

*Best.onnx => for nudity*

- Trained on: Porn and sexual images
- Format: Uses the ONNX format for model interoperability and deployment across different platforms.

*Superdupermodel.h5 => for violence*

- Trained on : Approximately 10-second video clips of sports fouls, MMA/Boxing, CCTV, and movie clips.
- Framework : Utilizes deep learning models with Keras, a high-level neural networks API. It is much heavier but allows for easier addition and configuration of training data.a
