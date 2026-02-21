import cv2

# Open the default camera (0 = webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()   # Read frame from camera
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    cv2.imshow("Original Video", frame)
    cv2.imshow("Grayscale Video", gray)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

# import cv2
# print(cv2.__version__)
