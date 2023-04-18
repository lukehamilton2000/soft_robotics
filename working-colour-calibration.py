import numpy as np
import cv2 as cv

def sorting(centers_initial, centers_present):

    '''this function sorts the centers detected
    '''
    centers_intermediate = np.ones((5, 2))
    # looping
    for i in range(5):
        for j in range(5):
            # calculating distance and judge
            print(centers_initial.shape, centers_present.shape)
            if np.sqrt(np.sum(np.square(centers_initial[i]-centers_present[j]))) < 40:
                centers_intermediate[i] = centers_present[j]
                break
    centers_intermediate = centers_intermediate.astype(np.int16)
    return centers_intermediate

if __name__ == '__main__':

    file_path = 'gripper-default-video-one.mov'
    #file_path = 'gripper-large-video-one.MOV'
    vc = cv.VideoCapture(file_path)

    if (vc.isOpened()== False):
        print("Error opening video stream or file")

    #origin = np.array([69, 471])
    origin = np.array([700, 100])
    shape = np.array([1920, 1080])

    centers_temp = np.array([])
    while (vc.isOpened()):
        ret, img_rgb = vc.read()

        if ret:
            cv.imshow('Original Video', img_rgb)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # cropping
        img_rgb = img_rgb[origin[0]:origin[0] + shape[0], origin[1]:origin[1] + shape[1]]


        # blurring
        img_rgb = cv.medianBlur(img_rgb, 9)

        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)

        # Convert to HSV color space
        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)

        # Filter for red color
        lower_red1 = np.array([0, 200, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv.inRange(img_hsv, lower_red1, upper_red1)
        mask2 = cv.inRange(img_hsv, lower_red2, upper_red2)
        mask = cv.bitwise_or(mask1, mask2)

        # Apply morphological operations to remove noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        mask = mask1 + mask2

        # detecting contours
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # finding centers of contours
        centers = []
        for cnt in contours:
            # Filter out small contours
            if cv.contourArea(cnt) < 100:
                continue
            (x, y), radius = cv.minEnclosingCircle(cnt)
            # Increase minimum radius
            if radius < 10:
                continue
            center = (int(x), int(y))
            cv.circle(img_rgb, center, 5, (255, 0, 0), -1)
            centers.append([center[0], center[1]])

        centers = np.array(centers)
        if len(centers_temp)==0:
            centers_temp = centers

        # sorting
        if len(centers_temp) > 0:
            centers = sorting(centers_temp, centers)
            centers_temp = centers

        #centers = sorting(centers_temp, centers)
        #centers_temp = centers
        sorted_centers = sorted(centers, key=lambda x: x[0])
        # drawing lines
        for i in range(len(sorted_centers) - 1):
            cv.line(img_rgb, (sorted_centers[i][0], sorted_centers[i][1]),
                    (sorted_centers[i + 1][0], sorted_centers[i + 1][1]), (255, 0, 0), 2)


        if ret:
            cv.imshow('image', cv.resize(img_rgb, (640, 360)))


            if cv.waitKey(27) & 0xFF == ord('q'):
                break

        else:
            break

    cv.destroyAllWindows()