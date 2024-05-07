_NOTE: BadCode contains different methods and building model and should be ingnored_


## Running the Program:
### Install Required Packages

Before running the application, you need to install the necessary packages listed in `requirements.txt`. Run the following command:

```bash
pip install -r requirements.txt
```
### Run the Video modration app:

```bash
python app.py
```

The program should run on the local servear and the website can be viewed on

 http://127.0.0.1:5000/

The order of the files is importent and will not work if changed
# The Program:
If you run the program correctky It shoudl look something like this:
<img src="Untitled.gif" alt="The Webpage">

### General Overview of the Modles:

*Best.onnx => for nudity*

- Trained on: Porn and sexual images
- Format: Uses the ONNX format for model interoperability and deployment across different platforms.

*Superdupermodel.h5 => for violence*

- Trained on : Approximately 10-second video clips of sports fouls, MMA/Boxing, CCTV, and movie clips.
- Framework : Utilizes deep learning models with Keras, a high-level neural networks. It is much heavier but allows for easier addition and configuration of training data.a
