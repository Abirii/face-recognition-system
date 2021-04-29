# Face recognition system

Based on [face_recognition](https://github.com/ageitgey/face_recognition) library 


### Upload your images and create dataset
1. Each image should have its directory, i.e single image in each directory
2. Place the images you want in the system under DATA/train 
###### NOTE: Each image should have only one person

```
/Data
  /train
     /person1
       person1.jpg
     /person2
        person1.jpg
     /person3
        person1.jpg
     ...
     ...
     ...
```
3. Create embedding dataset by open new py file and type
 ```
 import dataset
 import recognition
 # create npy.file
 dataset.create('embedding_data')
 # append all the images 
 recognition.append_from_directory()
 ```
 After npy file should be created names embedding_data.npy, 
 this file hold the embedding vectors(person_name, embedding) for each image in step 2

 
 ### recognition
 After steps 1-3 you can start recognition by type
 ```
  import recognition
  import utils
  # use 'hog' or 'cnn' for detection
  result = recognition.recognition_from_dataset('PATH_TO_IMAGE', detection_method='hog')
  utils.show_image(image=result, size=(1000,1000))
 ```
 press 'q' to close the resuls
 
 ### Requirements
  ```numpy  ``` 
  ```dlib  ``` 
  ```face_recognition ```  
  ```opencv ```
 
 
 
 
 
 #### Resources
 [face-recognition 1.3.0](https://pypi.org/project/face-recognition/)  

 [Face Recognition](https://github.com/ageitgey/face_recognition)

 
 
 
 

