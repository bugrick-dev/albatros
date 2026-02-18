import cv2
import numpy as np


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(gray, (11,11), 1)

    # canny = cv2.Canny(blur, 100, 150)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 120, 70]) 
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red_total = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_blue, mask_red_total)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)     
    
    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 1000:

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
            else:
                isim = "daire"
            
            cv2.putText(frame, isim, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            


    cv2.imshow("kamera", frame)
    cv2.imshow("maske", mask)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
