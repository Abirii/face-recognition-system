import cv2 as cv
import face_recognition
import os

def bgr_to_rgb(image):
    """
    Convert image from BGR (OpenCV ordering) to dlib ordering (RGB)
    :param image: Input image
    :return: RGB image
    """
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return rgb




def show_image(image, size=(250, 250)):
    """
    Show given face, press q for quit
    :param image: Input image
    :return: None
    """
    image = cv.resize(image, size)
    while True:
        cv.imshow("Output", image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break


def confirm_image(image_path, name):
    """
    Show an image and wait for confirm
    :param image:
    :return: Image select or not image select (True/False)
    """
    image = cv.imread(image_path)
    if image is None:
        print("Cannot load current image")
        return None

    image = cv.resize(image, (250, 250))
    print('Confirm image (y/n)', end=" ")

    while True:
        cv.imshow('Confirm: ' + name, image)

        key = cv.waitKey(1)
        # press y to confirm the image
        if key == ord('y'):
            print("yes")
            return True
        # press n to NOT confirm the image
        if key == ord('n'):
            print("no")
            return False



def bounding_box(image, coordinates, distance, name='Unknown'):
    """
    Drow bounding box
    :param image: input image
    :param coordinates: face coordinates
    :param distance: closet vector distance
    :return: image with bounding box around the faces
    """
    top, right, bottom, left = coordinates
    top -= 20
    left -= 20
    right += 10
    bottom += 10

    # if the face NOT recognized successfully, draw green and thick bounding box around the face
    if name == 'Unknown':
        image = cv.rectangle(image, (left-20, top-20), (right-10, bottom-10), (0, 255, 0), 2)


    # if the face recognized successfully draw a bounding box around the face with the name label below the face.
    else:
        image = cv.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)
        # draw a label with a name below the face
        cv.rectangle(image, (left, bottom - 20), (right, bottom), (0, 0, 255), cv.FILLED)
        cv.putText(image, name, (left + 6, bottom - 6), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        # draw a label with distance above
        cv.rectangle(image, (left, top - 8), (right, top+10), (0, 0, 255), cv.FILLED)
        cv.putText(image, str(distance), (left + 6, top+5), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    return image