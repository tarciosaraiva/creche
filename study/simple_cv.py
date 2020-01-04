import cv2
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread('%s/mandrill.png' % script_dir, cv2.IMREAD_GRAYSCALE)

while True:
    cv2.imshow('mandril', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
