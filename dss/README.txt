# Dead Sea Scrolls Task

# Structure

The folder contains a "requirements.txt" file, a "models" subfolder containing two neural networks. 
All necessary scripts to run the classification task are also included

Main folder includes:

- **char_seg.py**: by running this script the lines created by 
**lineseg.py** are segmented into separate characters.

- **classification.py**: by running this script the characters 
created by **char_seg.py** will be classified using the model created by **pretraining.py**

- **cleanDataGen.py**: by running this script, training data will be generated from the Habbakuk font which is used for pretraining the CNN in **pretraining.py**

- **lineseg.py**: by running this script images are segmented 
into separate lines

- **pretraining.py**: by running this script, a CNN is pretrained on the output from **cleanDataGen.py** and trained on the Monkbrill data

- **run_classify.sh**: running this script will run all other
scripts necessary for performing the classification 
(**lineseg.py** -> **char_seg.py** -> **classification.py**)

- **run_training.sh**: running this script will run all other
scripts necessary for performing model training
(**lineseg.py** -> **char_seg.py** -> **cleanDataGen.py** -> 
**pretraining.py** -> **classification.py**)

- **utils.py**: includes general functions used 
in the rest of the scripts

# Outputs
- **char_seg.py**: in each image's subfolder under the "lines" folder, 
a new "characters" subfolder will be created. "characters" will have 
subfolders of type "lineXX" (where "XX" represents a number) for each line.
These folders will include the images of the segmented characters for each line.

- **classification.py**: as many output .txt files as there are images in the original data, these .txt files contain the text as recognized by our program. 

- **cleanDataGen.py**: a new folder called "new_data" is created with training data generated from the Habbakuk font. 

- **lineseg.py**: a new "lines" folder is created. The folder contains subfolders 
named after each image. The subfolders contain the images of the separated lines.

- **pretraining.py**: 

# Running the scripts

## How to run the training and classification of the model
*This should be done when there is no model saved.*
- ./run_training.sh "folder_with_images"

## How to run the classification task without training 
*This can be used when you have a previously trained model saved.*

To run dataset from specified folder:

- ./run_classify.sh "folder_with_images"

To run default dataset ("data/image-data" required in main folder):

- ./run_classify.sh
- Results will be present in results/classification/

# Requirements
The **requirements.txt** file includes:

albumentations==1.1.0

ipython==8.4.0

matplotlib==3.5.2

networkx==2.8.3

numpy==1.22.4

pandas==1.4.2

Pillow==9.1.1

scikit_learn==1.1.1

scipy==1.8.1

tensorboard==2.9.0

tensorflow==2.9.1

tensorflow_datasets==4.6.0

tqdm==4.64.0


# Copyright
Scripts created by:

Subhaditya Mukherjee s4747925 && Isabelle Tilleman s3656586 && Leonidas Zotos s3396991 && Paul Pintea s3593673