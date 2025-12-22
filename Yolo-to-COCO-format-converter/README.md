# Yolo-to-COCO-format-converter

## How to use
### Requirements
- numpy
- OpenCV  

You can make same environment with anaconda environment.  
- `conda create -n Yolo-to-COCO python=3.8`  
- `conda activate Yolo-to-COCO`  
- `pip install numpy`  
- `pip install opencv-python`  
- `pip install imagesize`  

Just clone this repository.  
- `git clone https://github.com/alvarogm84-hub/sow3.git`  
- `cd Yolo-to-COCO-format-converter`  

### 1. Change `classes` with your own dataset.  
In `main.py`, there is a code that declare the classes. You will change this with your `obj.names`.  
Already changed for 12 classes Milestone dataset:

classes = [
    "Bicycle",
    "Motorcycle",
    "Car",
    "Van",
    "RV",
    "Single_Truck",
    "Combo_Truck",
    "Pickup_Truck",
    "Trailer",
    "Emergency_Vehicle",
    "Bus",
    "Heavy_Duty_Vehicle"
]


### 2. Prepare COCO annotation file from multiple YOLO annotation files.
#### Image and annotation files are in different directories 
Use this approach if your training data file structure looks like this:
<pre>
    dataset_root_dir_images/
        Photo_00001.jpg
        Photo_00002.jpg        
    dataset_root_dir_labels/
        Photo_00001.txt
        Photo_00002.txt
</pre>

You don't need to specify `yolo-subdir` argument.  


### 3. Run for train, val and test subsets
- `python main.py --path_images <Absolute path to dataset_root_dir_images> --path_annotations <Absolute path to dataset_root_dir_annotations> --output <Name of the json file>`  
- (For example train subset)`python main.py --path_images /..../Milestone/images/train --path_images /..../Milestone/labels/train --output /..../Milestone/annotations/train.json`

#### 3a Initial Yolo format Miletsone dataset structure
<pre>
    dataset_root_dir/
        images/
            train/
                Photo_00001.jpg
                Photo_00002.jpg   
            val/
                Photo_00001.jpg
                Photo_00002.jpg   
            test/
                Photo_00001.jpg
                Photo_00002.jpg   
        labels/
            train/
                Photo_00001.txt
                Photo_00002.txt
            val/
                Photo_00001.txt
                Photo_00002.txt
            test /       
                Photo_00001.txt
                Photo_00002.txt
</pre>

#### 3b Final COCO format Miletsone dataset structure
<pre>
    dataset_root_dir/
        annotations/
            train.json
            val.json
            test.json
        train/
            Photo_00001.jpg
            Photo_00002.jpg 
        val/
            Photo_00001.jpg
            Photo_00002.jpg 
        test/  
            Photo_00001.jpg
            Photo_00002.jpg 
</pre>

