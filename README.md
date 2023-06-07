# webcam-simle-detection-svm
detecte simle live on web cam using svm model trained on GENKI4


- the model has trained on [GENKI4 dataset](https://inc.ucsd.edu/mplab/398/) witch is 4000 pictures 
with two classes(smile and not smile)
- used opencv to and Scikit-image to detect faces and extract LBP and HOG features
- used Scikit-learn to fit svm model on data
- svm trained model is saved and uploaded in repository in order to use in live smile detection

### Installation
1. Create a virtual environment for the project:
```
python -m venv venv
```
2. Activate the virtual environment:
  - On Windows:
```
venv\Scripts\activate
```

  - On macOS and Linux:
```
source venv/bin/activate
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. run:
```
python main.py
```
