# Steps to reproduce
- docker pull msubhaditya/hwr_project
- docker run -i -t msubhaditya/hwr_project:version1
- cd hwr/
- Note: This is not supported on an M1 Mac
## Task 3 : IAM
- While you are still in the container, you can run the following commands to reproduce the results.
- cd /hwr/src/iam
### Testing
- For a folder of images. (Can just be run directly). The results will be saved in the folder results/iam_predictions as requested
  - python3 test.py --folder somefolder/
- For a single image. (Can just be run directly)
  - python3 test.py --imagepath someimage.jpg
- To run the evaluation on the validation or test set, you need to first have run the preprocessing step (1) that saves the dataset to the `temp_ds/` directory.
  - python3 test.py --evaluate

### Training if required
- To generate the train/test/val data -(1)
    - python3 datautils.py
- To run the actual training
    - python3 train.py temp_ds/train/
- The output models are saved in the folder conv_lstm_model/
