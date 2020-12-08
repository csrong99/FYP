import cv2
import numpy as np

# video = cv2.VideoCapture('http://192.168.0.169:8090/camera.mjpeg')
video = cv2.VideoCapture('http://192.168.0.169:8090/v4.h264')

while True:
    _, frame = video.read()

    ori_frame = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([110, 60, 0])
    upper_red = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Converting the image to black and white
    (threst, res) = cv2.threshold(res, 90, 255, cv2.THRESH_BINARY)

    _, contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    cv2.drawContours(frame, [cnt], 0, (255, 255, 0), 3)

    contours_2d = np.vstack(cnt.squeeze())

    # get the all index for xmin xmax ymin ymax
    xmin_contour = contours_2d[np.argmin(contours_2d[:,0]), :][0]
    xmax_contour = contours_2d[np.argmax(contours_2d[:,0]), :][0]
    ymin_contour = contours_2d[np.argmin(contours_2d[:,1]), :][1]
    ymax_contour = contours_2d[np.argmax(contours_2d[:,1]), :][1]

    cv2.rectangle(frame, (xmin_contour, ymin_contour), (xmax_contour, ymax_contour), (0, 0, 255), 3)

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    res = cv2.resize(res, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    ori_frame = cv2.resize(ori_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('ori', ori_frame)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
video.release()