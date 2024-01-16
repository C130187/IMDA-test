import argparse

import cv2
import os
from os.path import join

from imutils import paths
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from joblib import dump, load


class Captcha(object):
  def __init__(self):
    self.image_counts = {} # Count of images subsectioned and stored for each letter from original images
    self.input_dir = 'data/input'
    self.output_dir = 'data/output'
    self.training_data_dir = "data/preprocessed_characters"
    self.model_path = "model/captcha_model.h5"
    self.lb_path = "model/lb.pkl"

    os.makedirs(self.training_data_dir, exist_ok=True)

    if not (os.path.isfile(self.model_path) and os.path.isfile(self.lb_path)):
        print("Training model.....")
        self.train_and_save_model()

    self.model, self.lb = self.load_model()

  def prepare_training_data(self):
      print("Preparing training data")

      if len(os.listdir(self.training_data_dir))==0:
          input_files = os.listdir(self.input_dir)
          input_files_paths = [join(self.input_dir, f) for f in input_files]
          input_jpg_paths = [f for f in input_files_paths if f.endswith('.jpg')]
          input_files_paths = sorted([p for p in input_jpg_paths])
          output_files = os.listdir(self.output_dir)
          output_files_paths = sorted([join(self.output_dir, f) for f in output_files])

          for image_path, output_path in zip(input_files_paths, output_files_paths):
              f = open(output_path, 'r')
              output_str = f.read()
              f.close()

              letter_image_regions, gray = self.preprocess_and_return_regions(image_path)

              # Save each letter as a single image
              for i, letter_bounding_box in zip(output_str, letter_image_regions):
                  # Grab the coordinates of the letter in the image
                  x, y, w, h = letter_bounding_box

                  # Extract the letter from the original image with a 1-pixel margin around the edge
                  letter_image = gray[y - 1:y + h + 1, x - 1:x + w + 1]

                  # Get the folder to save the image in
                  save_path = os.path.join(self.training_data_dir, i)

                  # creating different output folder for storing different letters
                  if not os.path.exists(save_path):
                      os.makedirs(save_path)

                  # write the letter image to a file
                  count = self.image_counts.get(i, 1)

                  s = str(count) + "_" + image_path.split('/')[-1].split('.')[0] + ".png"
                  p = os.path.join(save_path, s)
                  cv2.imwrite(p, letter_image)

                  # increment the count
                  self.image_counts[i] = count + 1
      else:
          print("Data already present")

  def get_data_and_labels(self):
    print("Getting data and labels")

    self.prepare_training_data()
    #creating empty lists for storing image data and labels
    data = []
    labels = []
    for image in paths.list_images(self.training_data_dir):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (30,30))

        # adding a 3rd dimension to the image
        img = np.expand_dims(img, axis = 2)

        #grabing the name of the letter based on the folder it is present in
        label = image.split(os.path.sep)[-2]

        # appending to the empty lists
        data.append(img)
        labels.append(label)

    #converting data and labels to np array
    data = np.array(data, dtype = "float")
    labels = np.array(labels)

    print("Data length: ",len(data))
    print("Labels length: ", len(labels))

    #scaling the values of  data between 0 and 1
    data = data/255.0

    # Split the training data into separate train and test sets
    #(train_x, val_x, train_y, val_y) = train_test_split(data, labels, test_size=0.2, random_state=0)

    all_classes_data = set(labels)

    (train_x, val_x, train_y, val_y) = train_test_split(data, labels, test_size=0.2, random_state=0)

    missing_classes = {label for label in all_classes_data if len(train_y[train_y == label]) < 2}
    if missing_classes:
        # If any class is missing, move one additional instance of each missing class to the training set
        for missing_class in missing_classes:
            indices_to_move = [i for i, label in enumerate(val_y) if label == missing_class][:2]
            train_x = np.append(train_x, val_x[indices_to_move], axis=0)
            train_y = np.append(train_y, val_y[indices_to_move])
            val_x = np.delete(val_x, indices_to_move, axis=0)
            val_y = np.delete(val_y, indices_to_move)

    lb = LabelBinarizer().fit(train_y)
    train_y = lb.transform(train_y)
    val_y = lb.transform(val_y)

    print("Shapes")
    print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)

    return train_x, val_x, train_y, val_y, lb

  def train_and_save_model(self):
    print("Training and saving model and binariser")
    model2 = Sequential()
    model2.add(Flatten(input_shape=(30, 30, 1)))  # Flatten the 30x30 image
    model2.add(Dense(128, activation='relu'))  # Fully connected layer with ReLU activation
    model2.add(Dense(36, activation='softmax'))  # Output layer with softmax activation ,we have 10 digits + 26 letters = 36 classes
    model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    train_x, val_x, train_y, val_y, lb = self.get_data_and_labels()

    # Train the model with early stopping
    model2.fit(train_x, train_y, epochs=50, validation_data=(val_x, val_y), callbacks=[early_stopping])

    # Evaluate the model on the test set
    test_loss, test_accuracy = model2.evaluate(val_x, val_y)
    print(f"Test Accuracy: {test_accuracy}")

    # Save the trained model for future use
    model2.save(self.model_path)
    dump(lb, self.lb_path)

  def load_model(self):
    print("Loading model and binariser")
    model= load_model(self.model_path)
    lb = load(self.lb_path)
    return model, lb

  def preprocess_and_return_regions(self, im_path):

    image = cv2.imread(im_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold the image
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]

    # find the contours
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []

    # Now we can loop through each of the contours and extract the letter

    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        letter_image_regions.append((x, y, w, h))

    # Sort the detected letter images based on the x coordinate to make sure
    # we get them from left-to-right so that we match the right image with the right letter

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    return letter_image_regions, gray

  def __call__(self, im_path, save_path):
    """
    Algo for inference
    args:
    im_path: .jpg image path to load and to infer
    save_path: output file path to save the one-line outcome
    """
    # Creating an empty list for storing predicted letters
    predictions = []

    letter_image_regions, gray = self.preprocess_and_return_regions(im_path)

    # Save out each letter as a single image
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 1-pixel margin around the edge
        letter_image = gray[y - 1:y + h + 1, x - 1:x + w + 1]

        letter_image = cv2.resize(letter_image, (30,30))

        # Turn the single image into a 4d list of images
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # making prediction
        pred = self.model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = self.lb.inverse_transform(pred)[0]
        predictions.append(letter)

    # Print the captcha's text
    captcha_text_predicted = "".join(predictions)

    # Extract the directory from save_path
    directory = os.path.dirname(save_path)

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the result to the specified save_path
    with open(save_path, "w") as file:
        file.write(captcha_text_predicted)
    print(f"Result saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on a captcha image.")
    parser.add_argument("--im_path", type=str, help="Path to the input image (.jpg).", required=True)
    parser.add_argument("--save_path", type=str, help="Path to save the output result.", required=True)

    args = parser.parse_args()

    # Initialize the Captcha class
    captcha_model = Captcha()

    # Run inference
    captcha_model(args.im_path, args.save_path)

if __name__ == "__main__":
    main()
