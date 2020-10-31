# Introduction
**Problem:** 
Detect whether a patient have a begnin or malignant tumor based on his blood tests. More infos on the dataset [here](https://en.wikipedia.org/wiki/Fine-needle_aspiration).

**Solution:**
Implemented a [neural network (ANN)](https://en.wikipedia.org/wiki/Multilayer_perceptron) with Python 3 from scratch with two hidden layers and the [binary cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression) as cost function.

# Generate random training/test datasets
To generate random training and test datasets just run the evaluation.py script in the evaluate_model folder. 

*This script was provide in the subject, I did not code it,*
Generate a dataset by running the evaluation.py script gave in the subject. 
		(python3 evaluate_model/evaluation.py)
2. Train the model by running train script on the data_training set, it will generate a nn_weights.py file that will be used by the predict program after.
		(python3 multilayer_perceptron_train.py data_training.csv)
3. Predict the values with the predict script, the output is the result of the binary cross-entropy loss
		(python3 multilayer_perceptron_predict.py evaluate_model/data_test.csv)


## Train
The train script come with several options to better understand the result.
By default the program will print the TE (Training error) and VE (Validation error).

Training error = result of the binary cross-entropy applied on the training data (80% of the dataset given in arg)
Validation error = result of the binary cross-entropy applied on the validation data (20% of the dataset given in arg)

**You can find all the options by running the script with the arg -h. Here his some of them:**
[-pc] Print cost graph:
![enter image description here](https://i.ibb.co/p1Fchc9/multilayer-pc.png)

[-pr] Print result:
![enter image description here](https://i.ibb.co/8BgmQ3n/multilayer-pr.png)

**To run the script: 
python3 multilayer_perceptron_train.py data_training.csv**

## Predict
The training scripts generates a file with the weights in it (nn_weights.py). It's this file that will be use by the predict script.

**To run the script: 
python3 multilayer_perceptron_test.py data_test.csv**

The result is the error computed between the real value and my predicted one, it is computed by using the binary cross-entropy loss.
