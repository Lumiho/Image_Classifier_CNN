# GOAL:
    # Construct an image classifier using a Convolutional Neural Network (CNN).

# REQUIREMENTS:
    # Normalize the data (pixel values are floats [0,1])
    # 2D convolutional layer, 28 filters, 3x3 window size, ReLU activation
    # 2x2 max pooling
    # 2D convolutional layer, 56 filters, 3x3 window size, ReLU activation
    # fully-connected layer, 56 nodes, ReLU activation
    # fully-connected layer, 10 nodes, softmax activation

# NOTE:
    # Use the Adam optimizer, 32 observations per batch, and sparse categorical cross-entropy loss. Use the train and test splits provided # by fashion-mnist. Use the last 12000 samples of the training data as a validation set. Train for 10 epochs.

    # Print the number of trainable parameters in the model
    # Evaluate training and validation accuracy at the end of each epoch, and plot them as line plots on the same set of axes.
    # Evaluate accuracy on the test set.
    # Show an example from the test set for each class where the model misclassifies.
    # Comment on any other observations about the model performance
    # Resource: http://karpathy.github.io/2019/04/25/recipe/