# IAM Dataset Code

## Training procedure
- To generate the train/test/val data -(1)
    - python3 datautils.py
- To run the actual training
    - python3 train.py temp_ds/train/
- The output models are saved in the folder conv_lstm_model/
## Evaluation procedure
- For a single image. (Can just be run directly)
  - python3 test.py --imagepath someimage.jpg
- For a folder of images. (Can just be run directly). The results will be saved in the folder results/predictions.csv
  - python3 test.py --folder somefolder/
- To run the evaluation on the validation or test set, you need to first have run the preprocessing step (1) that saves the dataset to the `temp_ds/` directory.
  - python3 test.py --dataset temp_ds/test 
