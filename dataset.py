import numpy as np
import os


dataest_path = 'DATA/embedding_data.npy'


def create(dataset_name, shape=(0, 2)):
    '''
    Create new dataset (Train set) that store all embedding vectors npy file
    Save the npy under train directory
    :param dataset_name: data name
    :param shape: the shape of the data (0, 128)
    :return: None
    '''

    # save the npy under train directory
    path = os.path.join('DATA', dataset_name + '.npy')

    if os.path.isfile(path):
        print(dataset_name + " is already exists")
        return None
    else:
        # create an empty array
        new = np.empty(shape=shape)
        # save as npy file
        np.save(path, new)
        print(dataset_name + " was created")

        return None



def append(person_name, embedding):
    '''
    Append new image to the dataset
    :param name: name of  person
    :param embedding: embedding of person
    :param detection_method: the face detection method (cnn/hog)
    :return: true if the image confirm and added successfully, else false
    '''

    # APPEND 128-D VECTOR AND HIS CORRESPONDING NAME
    # Load the (npy) file into embedding_data
    data = np.load(dataest_path, allow_pickle=True)

    # create details of the person - numpy array of (person_name,embedding) in size (2,)
    # where the first value is the peron name and the secend is the embedding in size (128,)
    person_data = np.array([person_name, embedding], dtype=object)

    # append the person_data to the file
    data = np.vstack((data, person_data))
    # Save the update data as npy file
    np.save(dataest_path, data)

    print(person_name, 'is added')




def info():
    '''
    Print basic info
    :param data:
    :return:
    '''
    data = np.load(dataest_path, allow_pickle=True)
    print("npy file in shape", data.shape)
    if data.shape == (0, 2):
        print("Empty dataset")
        return None

    print("First value of each row is person name ->", type(data[0][0]))
    print("Second value of each row is embedding vector in size", data[0][1].shape, "->", type(data[0][1]))
    print("Number of persons:", data.shape[0])



def reset():
    '''
    Empty embedding dataet file (npy)
    :param data_path: path to dataset
    :return: None
    '''

    # load data
    data = np.load(dataest_path, allow_pickle=True)
    # save shape before delete
    prev_shape = data.shape
    # delete and save in empty
    empty = np.delete(data, np.arange(prev_shape[0]), axis=0)
    # over ride file
    np.save(dataest_path, empty)
    print(f'previous shape was {prev_shape}, and now {empty}')


def show_data():
    '''
    Print data
    :param data_path: data_path: path to dataset
    :return: None
    '''

    data = np.load(dataest_path, allow_pickle=True)
    print(data)


def names(print_flag=True):
    '''
    Get all the persons names
    :param data_path: data_path: path to dataset
    :return: list of all names
    '''

    data = np.load(dataest_path, allow_pickle=True)
    names = []
    for i in data:
        name = i[0]
        if print_flag == True:
            print(name)
        names.append(name)
    return names



def delete_person(name):
    '''
    Delete person from data set
    :param name: the name of the person
    :param data_path: data_path: path to dataset
    :return: true id deleted, else false
    '''
    if name not in names(print_flag=False):
        print(name + " not found")
        return False

    data = np.load(dataest_path, allow_pickle=True)
    for count, i in enumerate(data):
        if i[0] == name:
            data = np.delete(data, count, 0)
            np.save(dataest_path, data)
            print(name + " is deleted")
            return True


def get_embedding(name):
    '''
    Get an embedding of specific person
    :param name: name of the person
    :param data_path: data_path: path to dataset
    :return: embedding vector
    '''

    if name not in names(print_flag=False):
        print(name + " not found")

    data = np.load(dataest_path, allow_pickle=True)
    for i in (data):
        if i[0] == name:
            return i[1]



def get_row_data(index):
    '''
    Return row data by index
    :param index: index to return
    :return: name(<class 'str'>), embedding (<class 'numpy.ndarray'>)
    '''
    data = np.load(dataest_path, allow_pickle=True)
    row = data[index]
    return row[0], row[1]



