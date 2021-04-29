# Face recognition system

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
 dataset.create('embedding_data')
 ```
 After npy file should be created names embedding_data.npy, 
 this file hold the embedding vectors(person_name, embedding) for each image in step 2

 

 

