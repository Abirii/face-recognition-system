import face_recognition as fc
import dataset as ds
import cv2 as cv
import utils
import os

#dataest_path = 'DATA/embedding_data.npy'


def embedding(image_path, detection_method='hog'):
    '''
    Get an paht to image of SINGLE person and return his embedding
    :param image_path: path to SINGLE person image
    :param detection_method: face detection model to use: either hog or cnn
    '''
    # read the image
    image = cv.imread(image_path)
    if image is None:
        print("Cannot load current image")
        return None

    # convert images to RGB format
    rgb_frame = utils.bgr_to_rgb(image)

    # face_locations is a list of bounding boxes for each face location in a the image - list of [tuple(top, right, bottom, left).....]
    face_locations = fc.face_locations(rgb_frame, model=detection_method)

    # embedding is a list of 128-dimensional face encodings.
    # NOTE: Each embedding in the list is:  <class 'numpy.ndarray'> with shape of (128,) That store  <class 'numpy.float64'>
    embedding = fc.face_encodings(rgb_frame, face_locations)[0]
    return embedding



def recognition_from_dataset(image_path, detection_method='hog', threshold=0.6):
    '''
    Apply face recognition of the given face
    :param image_path: image path - SHOULD BE CROP FACE OF ONE PERSON
    :param tolerance: threshold value (default 0.6)
    :param embedding_data: npy file that hold the embeddings vector
    :param embedding_data_type: npy file that hold the embeddings vector (hog or cnn)
    :return: the name of the person in the image and distance, if not recognition will return "Unknown" and 1000 as distance
    '''
    # Loads an  image file (.jpg, .png, etc) into a numpy array in RGB format
    image = fc.load_image_file(image_path)
    # find all faces - face_locations is a list of bounding boxes for each face location in a the image [tuple(top, right, bottom, left)..]
    face_locations = fc.face_locations(image, model=detection_method)
    # embedding_list is a list of 128-dimensional face encodings.
    # NOTE: Each embedding in the list is:  <class 'numpy.ndarray'> with shape of (128,) That store  <class 'numpy.float64'>
    embedding_list = fc.face_encodings(image, face_locations)

    # run over all the faces that detected and compare each face to ALL stored face in the dataset
    for locations_index, unknown_embedding in enumerate(embedding_list):

        # initialize default min. distance and default name
        closet_name, min_distance = 'Unknown', 1000
        # run over the embedding dataset
        for i in range(len(ds.names(print_flag=False))):
            name, known_embedding = ds.get_row_data(index=i)
            # Compare the unknow_embedding to every embedding vector in the dataset by euclidean distance.
            # For each comparison the distance tells you how similar the faces are.
            # NOTE - The parameters for the function are:
            # face_encodings – List of face encodings to compare, so we need to use [] (list of)
            # face_to_compare – A face encoding to compare against
            # distance is store in numpy ndarray with the distance for each face in the same order as the ‘faces’ array, but we compare 1:1 so we intrested in the first index only.
            distance = fc.face_distance(face_encodings=[known_embedding], face_to_compare=unknown_embedding)[0]

            # The aim of the code block is to search the min. distance and store the vector and his corresponding name.
            # First distance should be < tolerance
            if distance <= threshold:
                # Check if the distance is the min. distance until this point.
                if distance <= min_distance:
                    min_distance = distance
                    closet_name = name

        # draw on the given image (BGR format)
        image = utils.bounding_box(image, face_locations[locations_index], min_distance, closet_name)


    # convert to RGB back
    return cv.cvtColor(image, cv.COLOR_RGB2BGR)



def verify(emb1, emb2, threshold=0.6):
    '''
    get two embedding vectors, if their distance is smaller than threshold return "same" else "different"
    :param emb1: first embedding vector
    :param emb2: second embedding vector
    :param threshold: the threshold to verify as same person
    :return: true or false and the distance
    '''

    # face_distance- For each comparison the distance tells you how similar the faces are.
    # NOTE - The parameters for the function are:
    # face_encodings – List of faces encodings to compare, so we need to use [] (list of)
    # face_to_compare – A face encoding to compare against
    # distance store a numpy ndarray with the distance for each face in the same order as the ‘faces’ array, but we
    # compare 1:1 so we interested in the first index only.
    distance = fc.face_distance(face_encodings=[emb1], face_to_compare=emb2)[0]

    if distance <= threshold:
        return True, distance
    return False, distance




def append_from_directory(dir_path='DATA/train', detection_method='hog', confirm_flag=False):
    '''
    Append all the imgaes from train set
    :param dir: path to embedding dataset
    :return:
    '''

    flag = True
    for dir, dirname, filename in os.walk(dir_path):

        # skip first file
        if flag is True:
            flag = False
            continue
        # get full path to an image in train the set
        full_image_path = os.path.join(dir, filename[0])

        # extract peron name from file name
        name = filename[0].split('.')[0] # remove ".*" from file
        name = name.replace('_', " ") # change "_" to " "

        # confirm image before adding
        if confirm_flag:
            confirm = utils.confirm_image(full_image_path, name)
            if confirm:
                # get embedding
                new_embedding = embedding(image_path=full_image_path, detection_method=detection_method)
                # add embedding to dataset
                ds.append(person_name=name, embedding=new_embedding)

        # not need to confirm
        else:
            # get embedding
            new_embedding = embedding(image_path=full_image_path, detection_method=detection_method)
            # add embedding to dataset
            ds.append(person_name=name, embedding=new_embedding)









