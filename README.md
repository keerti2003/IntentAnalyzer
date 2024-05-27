
# Intention Analyzer App

This repository contains the code for a text classification model using a Convolutional Neural Network (CNN) to predict the intent of a given sentence. The project uses pre-trained GloVe word embeddings for text representation and Keras for building and training the model.

## Project Structure

- `training_code.py`: Script for training the CNN model.
- `test_code.py`: Streamlit app for testing the trained model.
- `data/`: Directory containing processed training and testing data (`train_text.npy`, `train_label.npy`, `test_text.npy`, `test_label.npy`).
- `model.h5`: The trained Keras model (not included in the repository, generated after training).
- `label_encoder.pkl`: Label encoder for transforming labels (generated after training).
- `tokenizer.pkl`: Tokenizer for text processing (generated after training).

## Requirements

To run the training and testing scripts, you need to have the following libraries installed:

- numpy
- matplotlib
- keras
- tensorflow
- sklearn
- joblib
- streamlit

## Data

The training and testing data should be in the form of `.npy` files containing texts and their corresponding labels. Place these files in the `data/` directory.

## GloVe Embeddings

The project uses pre-trained GloVe embeddings. Download the `glove.6B.100d.txt` file from [this link](https://nlp.stanford.edu/projects/glove/) and place it in a directory named `glove.6B/` within the base directory of the project.

## Training the Model

To train the model, run the `train.py` script. This will:

- Load and preprocess the training data.
- Encode the labels.
- Tokenize the texts.
- Create an embedding matrix using GloVe embeddings.
- Build and train the CNN model.
- Save the trained model, label encoder, and tokenizer.

```sh
python train.py
```

## Running the Streamlit App

To test the trained model using a web interface, run the `test.py` script. This will start a Streamlit app where you can enter a sentence and get the predicted intent.

```sh
streamlit run test.py
```

## Usage

1. Ensure you have the required libraries installed.
2. Download the GloVe embeddings and place the file in the correct directory.
3. Prepare your training and testing data in `.npy` format and place them in the `data/` directory.
4. Run the `train.py` script to train the model.
5. Run the `test.py` script to start the Streamlit app and test the model.


## Acknowledgments

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Streamlit](https://streamlit.io/)

## Contact

For any inquiries or issues, please open an issue in the repository or contact me at [my mail](mailto:keerti2003.mk@gmail.com)


