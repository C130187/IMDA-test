# IMDA-test

### Step 1 : Clone the git repository
`git clone https://github.com/C130187/IMDA-test.git`

### Step 2 : Navigate to project directory
`cd IMDA-test`

### Step 3 : Create virtual environment with python3.11
`python3 -m venv venv`
Activate the virtual environment:

**On Windows:**
`.\venv\Scripts\activate`

**On Unix or MacOS:**
`source venv/bin/activate`

### Step 4 : Install dependencies using:
`pip3 install -r requirements.txt`

### Step 5: Run inference on the captcha model 

Run the python file captcha-converter.py with the following arguments:
--im_path = Path to the input image (.jpg)
--save_path = Path to save the output result

Example:
To run the project, execute the following command:

`python captcha-converter.py --im_path='data/test_input/input100.jpg' --save_path='data/test_output/output100.txt'`

## Project Structure

**data/input** - sample captcha images from provided link in the test, everything except input100.jpg which is later used for testing

**data/output** - output string text files for the images in data/input. output21.txt was missing so I added that myself.

**preprocessed_characters** - preprocessing the sample captcha images in data/input, splitting each image into 5 subsections, each containing one digit or letter
grouping these subsections by their corresponding digit or letter and storing it here for training purposes

**data/test_input** - path for sample test images

**data/test_output** - path for test output

**model** - contains saved captcha model and LabelBinariser used to map letter/digit to binary for training, and to map back from binary to letter/digit at inference

