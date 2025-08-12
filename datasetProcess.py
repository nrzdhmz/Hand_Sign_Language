import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
fixed_size = 400

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        h_img, w_img, _ = img.shape

        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, w_img)
        y2 = min(y + h + offset, h_img)

        imgCrop = img[y1:y2, x1:x2]

        red_bg = np.zeros((fixed_size, fixed_size, 3), np.uint8)
        red_bg[:] = (0, 0, 255)

        h_crop, w_crop = imgCrop.shape[:2]

        scale = fixed_size / max(h_crop, w_crop)
        new_w, new_h = int(w_crop * scale), int(h_crop * scale)
        imgResize = cv2.resize(imgCrop, (new_w, new_h))

        x_offset = (fixed_size - new_w) // 2
        y_offset = (fixed_size - new_h) // 2

        red_bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = imgResize

        cv2.imshow("imgCrop", red_bg)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
