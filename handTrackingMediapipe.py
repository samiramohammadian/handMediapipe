import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Define coordinates for dividing the image into four parts
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
part_width = int(width / 2)
part_height = int(height / 2)
lines = [[[part_width, 0], [part_width, height]], [[0, part_height], [width, part_height]]]
rectangles = [[(0, 0), (part_width, part_height)], [(part_width, 0), (width, part_height)],
              [(0, part_height), (part_width, height)], [(part_width, part_height), (width, height)]]

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)

        # Process the image with MediaPipe Hands
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw lines on each part
        for line in lines:
            cv2.line(image, tuple(line[0]), tuple(line[1]), (0, 0, 0), 1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Find the part with the maximum landmarks

                
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 250, 250), thickness=2, circle_radius=2),)
                
                max_landmarks = 0
                max_rectangle_index = -1
                for i, rectangle in enumerate(rectangles):
                    rectangle_x1, rectangle_y1 = rectangle[0]
                    rectangle_x2, rectangle_y2 = rectangle[1]
                    num_landmarks = 0
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        if rectangle_x1 <= x <= rectangle_x2 and rectangle_y1 <= y <= rectangle_y2:
                            num_landmarks += 1
                    if num_landmarks > max_landmarks:
                        max_landmarks = num_landmarks
                        max_rectangle_index = i

                # Increase line width for the part with the most landmarks
                if max_rectangle_index != -1:
                    rectangle_x1, rectangle_y1 = rectangles[max_rectangle_index][0]
                    rectangle_x2, rectangle_y2 = rectangles[max_rectangle_index][1]
                    cv2.rectangle(image, (rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2),(255, 255, 255), 15)
                    cv2.rectangle(image, (rectangle_x1 , rectangle_y1), (rectangle_x2, rectangle_y2),(225, 225, 225), 10)
                    cv2.rectangle(image, (rectangle_x1 , rectangle_y1), (rectangle_x2, rectangle_y2),(200, 200, 200), 5)

                        # Check if thumb is up
                    thumb_is_up = (
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                        < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    )

                    # Add text to the image indicating whether the thumb is up or down
                    if thumb_is_up == True:
                        cv2.putText(
                        image,
                        "thumb is up.... do things",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255) if thumb_is_up else (0, 0, 255),
                        2,
                    )
                    else :
                        cv2.putText(
                        image,
                        "thumb is down.... do things",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255) ,
                        2,
                    )
                    



        # Show the image with lines and rectangles
        cv2.imshow('MediaPipe Hands', image)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()