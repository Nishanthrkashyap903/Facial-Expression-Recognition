from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomCrop
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import configuration as cfg
from early_stopping import EarlyStopping
from lr_scheduler import LRScheduler
from torchvision import transforms
from cnn import EmotionVGG
from torchvision import datasets
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import SGD
import torch.nn as nn
import argparse
import torch
import math

def createTransforms(): 
    # Create a list of transformations to apply on each image during
    # training/validation and testing

    train_transform = transforms.Compose([
        Grayscale(num_output_channels=1),
        RandomHorizontalFlip(),
        RandomCrop((48, 48)),
        ToTensor()
    ])
    
    test_transform = transforms.Compose([
        Grayscale(num_output_channels=1),
        ToTensor()
    ])

    return train_transform, test_transform

def loadData(train_transform, test_transform):
    # Load the datasets and apply data augmentation.
    train_data = datasets.ImageFolder(cfg.TRAIN_DIR, transform=train_transform)
    test_data = datasets.ImageFolder(cfg.TEST_DIR, transform=test_transform)

    # Extract the class labels and the total number of classes
    classes = train_data.classes

    # Split the training dataset into a training set and a validation set. 
    train_size = math.floor(len(train_data) * cfg.TRAIN_SIZE)
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    # Change the validation transformation. 
    val_data.dataset.transforms = test_transform

    # Load our own dataset and store each sample with their corresponding labels
    train_dataloader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE)
    val_dataloader = DataLoader(val_data, batch_size=cfg.BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=cfg.BATCH_SIZE)

    return train_dataloader, val_dataloader, test_dataloader, classes, test_data

# Plot the training history. 
def plotHistory(history, save_plot): 
    # Set the style for the plot
    plt.style.use("seaborn-v0_8-poster")

    # Draw the plot
    plt.figure()
    plt.plot(history["train_acc"], label="Training accuracy")
    plt.plot(history["val_acc"], label="Validation accuracy")
    plt.plot(history["train_loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.ylabel("Loss/Accuracy")
    plt.xlabel("Number of Epochs")
    plt.title("Training loss and accuracy on FER2013")
    plt.legend(loc = "upper right")
    plt.savefig(save_plot)

def train(args):
    # Use CUDA is available. 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Current training device: {device}")

    # Create a list of transformations during training/validation and testing. 
    train_transform, test_transform = createTransforms()
    
    # Load the data from the directories
    train_dataloader, val_dataloader, test_dataloader, classes, test_data = loadData(train_transform, test_transform)

    # Number of classes
    num_classes = len(classes)
    print(f"[INFO] Class labels: {classes}")

    # Initialize the model and send it to device
    # TODO: Pass the model name from the main function. 
    model = EmotionVGG(in_channels=1, num_classes=num_classes, model_name=args.model)
    model = model.to(device)
    
    # Initialize our optimizer and loss function
    optimizer = SGD(params=model.parameters(), lr=cfg.LEARING_RATE)
    loss_func = nn.CrossEntropyLoss()
    
    # Initialize the learning rate scheduler and early stopping mechanism
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()
    
    # Save the training history. 
    history = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": []
    }

    # Iterate through the epochs
    print(f"[INFO] Start training...")
    start_time = datetime.now()
    
    for epoch in range(1, args.num_epochs + 1):
    
        print(f"[INFO] Training epoch: {epoch}/{args.num_epochs}")
    
        ######### TRAIN THE MODEL #########
        # Set the model to training mode
        model.train()
    
        # Initialize the total training and validation loss and the total number of correct predictions in both steps
        total_train_loss = 0
        total_val_loss = 0
        train_acc = 0
        val_acc = 0
    
        # Iterate through the training set
        for (data, target) in train_dataloader:
            # move the data into the device used for training,
            data, target = data.to(device), target.to(device)
    
            # Create the predictions from the model. 
            predictions = model(data)

            # Calculate the loss and add it to the total
            loss = loss_func(predictions, target)
            total_train_loss += loss

            # Zero the gradients accumulated from the previous operation,
            optimizer.zero_grad()

            # Perform a backward pass
            loss.backward()

            # Update the model parameters
            optimizer.step()
    
            # Keep track of the number of correct predictions. 
            train_acc += (predictions.argmax(1) == target).type(torch.float).sum().item()
    
        ########## VALIDATION MODE ##########
        model.eval()  # Go to evaluation mode. 
    
        # prevents pytorch from calculating the gradients, reducing
        # memory usage and speeding up the computation time (no back prop)
        with torch.set_grad_enabled(False):
            for (data, target) in val_dataloader:
                # move the data into the device used for testing
                data, target = data.to(device), target.to(device)
    
                # perform a forward pass and calculate the training loss
                predictions = model(data)
                loss = loss_func(predictions, target)
    
                # add the training loss and keep track of the number of correct predictions
                total_val_loss += loss
                val_acc += (predictions.argmax(1) == target).type(torch.float).sum().item()
        
        # Calculate the steps per epoch for training and validation set
        train_steps = len(train_dataloader.dataset) // cfg.BATCH_SIZE
        val_steps = len(val_dataloader.dataset) // cfg.BATCH_SIZE

        # Calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
    
        # calculate the train and validation accuracy
        train_acc = round(train_acc / len(train_dataloader.dataset), 2)
        val_acc = round(val_acc / len(val_dataloader.dataset), 2)
    
        # print model training and validation records
        print(f"Train loss: {round(avg_train_loss.item(), 2)}  .. Train accuracy: {train_acc}")
        print(f"Validation loss: {round(avg_val_loss.item(), 2)}  .. Validation accuracy: {val_acc}", end='\n\n')

        # Save the training history
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss.cpu().detach().numpy())
        history["val_acc"].append(val_acc)
    
        # Execute the learning rate scheduler and early stopping
        validation_loss = avg_val_loss.cpu().detach().numpy()
        lr_scheduler(validation_loss)
        early_stopping(validation_loss)
    
        # stop the training procedure due to no improvement while validating the model
        if early_stopping.early_stop_enabled:
            break
    
    end_time = datetime.now()
    print(f"[INFO] Total training time: {end_time - start_time}...")

    # Save the trained model to disk
    if device == "cuda":
        model = model.to("cpu")
    torch.save(model.state_dict(), f"models/{args.model}.pt")
    
    # Plotting the history
    plotHistory(history, f"acc_and_loss_{args.model}.png")

    ################## TEST TRAINED MODEL ##################### 
    # Mode the model to the device.
    model = model.to(device)
    with torch.set_grad_enabled(False):
        # Evaluation mode.
        model.eval()
    
        # Keep track of our predictions
        predictions = []
    
        for (data, _) in test_dataloader:
            # Move the data into the device used for testing
            data = data.to(device)
    
            # Perform a forward pass and calculate the training loss
            output = model(data)
            output = output.argmax(axis=1).cpu().numpy()
            predictions.extend(output)
 
    # evaluate the network
    print("[INFO] Evaluating trained model...")
    actual = [label for _, label in test_data]
    print(classification_report(actual, predictions, target_names=test_data.classes))

    

def main(): 
    # initialize the argument parser and establish the arguments required
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="VGG11", type=str, help="Specify the model name. Choose from this list: [VGG11, VGG13, VGG16, VGG19]")
    parser.add_argument('-e', '--num-epochs', default=25, type=int, help="The number of epochs needed to train the model")
    args = parser.parse_args()
    # print(args)
    train(args)

if __name__=="__main__": 
    main()