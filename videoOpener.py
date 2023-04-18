import cv2

# create a VideoCapture object
cap = cv2.VideoCapture('gripper-default-video-one.mov')

# check if the file was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# loop through the video frames
while cap.isOpened():
    # read a frame
    ret, frame = cap.read()

    # check if the frame was successfully read
    if not ret:
        break

    # display the frame
    cv2.imshow('frame', frame)

    # wait for a key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
