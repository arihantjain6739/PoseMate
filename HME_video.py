import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set the video source: 0 for webcam
video_source = 0  
# Replace with your video file
cap = cv2.VideoCapture(video_source)

# Set video resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Pose
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video capture ended.")
            break

        # Flip the frame horizontally (optional for mirror-like view)
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw landmarks and connections on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmark style
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Connection style
            )

        # Resize frame to fit the screen (optional)
        display_frame = cv2.resize(frame, (960, 540))

        # Display the processed frame
        cv2.imshow('Pose Estimation', display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
