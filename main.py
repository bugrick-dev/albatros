import cv2
import numpy as np

def empty (a):
    pass


cap = cv2.VideoCapture(0)


cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 300, 250)

cv2.createTrackbar("H Min", "Settings", 100, 179, empty)
cv2.createTrackbar("H Max", "Settings", 140, 179, empty)
cv2.createTrackbar("S Min", "Settings", 120, 255, empty)
cv2.createTrackbar("S Max", "Settings", 255, 255, empty)
cv2.createTrackbar("V Min", "Settings", 255, 255, empty)
cv2.createTrackbar("V Max", "Settings", 255, 255, empty)
    


# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (320, 240))


    if not ret:
        break

    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(gray, (11,11), 1)

    # canny = cv2.Canny(blur, 100, 150)
    
    h_min = cv2.getTrackbarPos("H Min", "Settings")
    h_max = cv2.getTrackbarPos("H Max", "Settings")
    s_min = cv2.getTrackbarPos("S Min", "Settings")
    s_max = cv2.getTrackbarPos("S Max", "Settings")
    v_min = cv2.getTrackbarPos("V Min", "Settings")
    v_max = cv2.getTrackbarPos("V Max", "Settings")    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #h, s, v = cv2.split(hsv)


    

    #v_balanced = clahe.apply(v)

    #hsv_new = cv2.merge([h, s, v_balanced])

    lower_blue = np.array([h_min, s_min, v_min]) 
    upper_blue = np.array([h_max, s_max, v_max])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red1 = np.array([0, s_min, v_min])
    upper_red1 = np.array([10, s_max, v_max])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, s_min, v_min])
    upper_red2 = np.array([180, s_max, v_max])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red_total = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_blue, mask_red_total)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)     
    


    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 500:

            perimeter = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)

            corners = len(approx)

            x, y, w, h = cv2.boundingRect(approx)
            
            aspectRatio = float(w) / h


            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)



            if corners == 3:
                isim = "ucgen"
            elif corners == 4:
                if 0.95 < aspectRatio < 1.05:
                    isim = "kare"
                else:
                    isim = "dikdortgen"
            elif corners == 6:
                isim = "hexagon"
            else:
                isim = "circle"
            
            cv2.putText(frame, isim, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            


    cv2.imshow("kamera", frame)
    cv2.imshow("maske", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
