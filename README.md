# Facial-Expression-Recognition
A Machine Learning system that can detect facial emotions on human faces.

## Description
Implement a CNN model to detect facial emotions on human faces. 

## Getting Started

### Dependecies
* Install all project dependecies using pip command
```
pip3 install -r requirements.txt
```

### Dataset
* Step 1: Sign up for Kaggle. 
* Step 2: Get Kaggle API credentials. 
    * Go to "Account Setting". 
    * Scroll down to the section labeled "API" and click on the "Create New API Token" button.
    * This will download a file named "kaggle.json" containing your API credentials.
    * Keep "kaggle.json" in "~/.kaggle/" to be able to use Kaggle API. 
* Step 3: Download the dataset using Kaggle API.
```
kaggle datasets download -d msambare/fer2013
```
* Step 4: Unzip the dataset.
```
unzip fer2013.zip
```
* Step 5: Combine the "test" and "train" folders into a folder called "datasets" 

### Dependecies
* Install all project dependecies using pip command
```
pip3 install -r requirements.txt
```

### Train the models
* Step 1: Choose the device to run the model
    * Type ```nvidia-smi``` to see all available GPUs in the CS machine. 
    * Choose the one that is not being used by other processes. 
* Step 2: Run the following command to train the model 
```
CUDA_VISIBLE_DEVICES = <gpu_number> python3 train.py --model <model_name> --num-epochs <number of epochs>
```
* Step 3: After the training, the model will save its checkpoint inside the ```models``` folder and the accuracy vs loss plot
under the name ```acc_and_loss_<model_name>.png```

## Authors

Contributors names and contact info
* Bao Nguyen (btn6364@rit.edu) 
* Neha Kulkarni  