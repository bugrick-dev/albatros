from picamera2 import Picamera2
import cv2
import numpy as np

def empty (a):
    pass


picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 300, 250)

cv2.createTrackbar("H Min", "Settings", 100, 179, empty)
cv2.createTrackbar("H Max", "Settings", 140, 179, empty)
cv2.createTrackbar("S Min", "Settings", 90, 255, empty)
cv2.createTrackbar("S Max", "Settings", 255, 255, empty)
cv2.createTrackbar("V Min", "Settings", 50, 255, empty)
cv2.createTrackbar("V Max", "Settings", 255, 255, empty)
    


# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    
    frame = picam2.capture_array()
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(gray, (11,11), 1)

    # canny = cv2.Canny(blur, 100, 150)
    
    h_min = cv2.getTrackbarPos("H Min", "Settings")
    h_max = cv2.getTrackbarPos("H Max", "Settings")
    s_min = cv2.getTrackbarPos("S Min", "Settings")
    s_max = cv2.getTrackbarPos("S Max", "Settings")
    v_min = cv2.getTrackbarPos("V Min", "Settings")
    v_max = cv2.getTrackbarPos("V Max", "Settings")    


    #h, s, v = cv2.split(hsv)


    

    #v_balanced = clahe.apply(v)

    #hsv_new = cv2.merge([h, s, v_balanced])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([h_min, s_min, v_min]) 
    upper_blue = np.array([h_max, s_max, v_max])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5,5), np.uint8)
    mask_blue = cv2.erode(mask_blue, kernel, iterations=2)

    mask = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    
    lower_red1 = np.array([0, s_min, v_min])
    upper_red1 = np.array([10, s_max, v_max])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    mask_red1 = cv2.erode(mask_red1, kernel, iterations=2)

    lower_red2 = np.array([170, s_min, v_min])
    upper_red2 = np.array([180, s_max, v_max])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red2 = cv2.erode(mask_red2, kernel, iterations=2)

    mask_red_total = cv2.bitwise_or(mask_red1, mask_red2)

    mask = cv2.bitwise_or(mask_blue, mask_red_total)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    


    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 500:

            perimeter = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)

            corners = len(approx)

            x, y, w, h = cv2.boundingRect(approx)
            
            aspectRatio = float(w) / h

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Merkeze bir nokta ve koordinatları yazalım
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"X: {cx} Y: {cy}", (x, y + h + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)



            if corners == 3:
                isim = "ucgen"
            elif corners == 4:
                if 0.90 < aspectRatio < 1.10:
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


picam2.stop()
cv2.destroyAllWindows()
