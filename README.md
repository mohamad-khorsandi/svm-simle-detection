# webcam-simle-detection-svm
detect simple live on webcam using SVM model trained on GENKI4


- the model has trained on [GENKI4 dataset](https://inc.ucsd.edu/mplab/398/) which is 4000 pictures 
with two classes(smile and not smile)
- SVM model train
  - used opencv and Scikit-image to detect faces and extract LBP and HOG features
  - used Scikit-learn to fit a SVM model on data
- CNN models train

  - used MobileNetV2 with pre-trained weights without classifier layers
  - classifier layer added to the model
  - all weights trained for this problem
- SVM trained model is saved and uploaded to the repository in order to use in live smile detection 

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
