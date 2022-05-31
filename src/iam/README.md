# IAM Dataset Code

## Training procedure

## Evaluation procedure
- To run the evaluation on the validation, you need to first have run the preprocessing step that saves the dataset to the `temp_ds/` directory.
  - python3 test.py conv_lstm_model --dataset temp_ds/test 
- For a single image
  - python3 test.py conv_lstm_model --imagepath someimage.jpg