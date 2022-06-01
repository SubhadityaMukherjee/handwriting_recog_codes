# Steps to reproduce
- docker run -i -t group3hwr/hwrproject:1.0.0 bash && cd hwr/
## Task 3 : IAM
- While you are still in the container, you can run the following commands to reproduce the results.
- cd /hwr/src/iam
### Testing
- For a folder of images. (Can just be run directly). The results will be saved in the folder results/iam_predictions as requested
  - python3 test.py --folder somefolder/
- For a single image. (Can just be run directly)
  - python3 test.py --imagepath someimage.jpg
- To run the evaluation on the validation or test set, you need to first have run the preprocessing step (1) that saves the dataset to the `temp_ds/` directory.
  - python3 test.py --dataset temp_ds/test 

### Training if required
- To generate the train/test/val data -(1)
    - python3 datautils.py
- To run the actual training
    - python3 train.py temp_ds/train/
- The output models are saved in the folder conv_lstm_model/


## Useful
- IAM Data: https://keras.io/examples/vision/handwriting_recognition/
- https://pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/

## Character segmentation Papers
- https://rug.on.worldcat.org/atoztitles/link?sid=google&auinit=%C3%98D&aulast=Trier&atitle=Feature+extraction+methods+for+character+recognition-a+survey&id=doi:10.1016/0031-3203(95)00118-2&title=Pattern+Recognition&volume=29&issue=4&date=1996&spage=641
- https://rug.on.worldcat.org/atoztitles/link?sid=google&auinit=RG&aulast=Casey&atitle=A+survey+of+methods+and+strategies+in+character+segmentation&id=doi:10.1109/34.506792&title=IEEE+Transactions+on+Pattern+Analysis+and+Machine+Intelligence&volume=18&issue=7&date=1996&spage=690&issn=0162-8828