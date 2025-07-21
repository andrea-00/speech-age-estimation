"""
This is the main training script for the age recognition model.

It performs the following steps:
1.  Loads the dataset metadata from the .csv file.
2.  Splits the data into training and testing sets.
3.  Defines and builds the full scikit-learn pipeline, including the custom
    feature extractor and the regressor/classifier.
4.  Trains the pipeline on the training data.
5.  Evaluates the model's performance on the test set.
6.  Saves the final, trained pipeline to the 'models/' directory.
"""