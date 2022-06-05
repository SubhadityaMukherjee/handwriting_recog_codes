if [ -z "$1" ]
then
    echo "No custom folder, running on given data"
    echo "Line Segmentation"
    python3 lineseg.py
else
    echo "Custom folder, running on custom data"
    python3 lineseg.py -f $1
fi
echo "Character Segmentation"
python3 char_seg.py

python3 cleanDataGen.py
python3 pretraining.py
python3 classification.py