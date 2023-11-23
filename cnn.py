import torch.nn as nn

class EmotionVGG(nn.Module):
    model_names = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, in_channels = 1, num_classes = 7, model_name = "VGG11"):
        super(EmotionVGG, self).__init__()
        if model_name not in self.model_names: 
            raise ValueError(f"Invalid model name: {model_name}. Must choose one of the these models: {list(self.model_names.keys())}")
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Create the convolution layers. 
        self.conv_layers = self.create_conv_layers(self.model_names[self.model_name])

        # Create the fully-connected layers 
        # Flatten -> 128 x 128 x 7 Linear layers. 
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(512 * 1 * 1, 128), # TODO: Can we replace 128 with another number? What's the difference?
            nn.ReLU(), 
            nn.Dropout(p=0.5), # This layer is not in the orginal paper. 
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(128, num_classes)
        )

    def __str__(self):
        return self.model_name
    
    # Generate the convolutional layers within the network
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture: 
            if x != "M": # Convolution layer
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1, stride=1))
                layers.append(nn.BatchNorm2d(x)) # This layer is not in the original paper
                layers.append(nn.ReLU())
                in_channels = x
            else: # Max pooling layer 
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    # Implement the forward pass that takes a batch of images (x) and produces a batch of logits
    def forward(self, x):
        # Get the output from the convolution layer. 
        out = self.conv_layers(x)
        
        # Flatten the image from (batch_size, 512, 1, 1) to (batch_size, 512)
        out = out.reshape(out.shape[0], -1)

        # Pass the flatten image to the fully connected layer
        out = self.fully_connected_layers(out)

        return out
 
 
    