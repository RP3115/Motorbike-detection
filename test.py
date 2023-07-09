import cv2

image=cv2.imread("example.png")
cv2.imshow("op1",image)
img=image[20:40, 50:100]
cv2.imshow("op2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()