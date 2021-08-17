import cv2
import sys
import imutils
# writer = cv2.VideoWriter(args[1],cv2.VideoWriter_fourcc(*'MJPG'), 10, (600,600))
videoPath   = sys.argv[1]
video       = cv2.VideoCapture(videoPath)
cascPath    = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
check, frame = video.read()
if check == False:
    print('video Not Found. Please Enter a Valid Path (Full path of video Should be Provided).')

print('Detecting people...')


while video.isOpened():
    # Capture frame-by-frame
    check, frame = video.read()
    frame = imutils.resize(frame, width = min(800, frame.shape[1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()