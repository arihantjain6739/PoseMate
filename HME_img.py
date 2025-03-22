import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load an image
image_path = "stand.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load the image. Please check the image path.")
else:
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image_rgb)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        print("Pose landmarks detected!")

        # Extract and print landmark data
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

        # Draw keypoints on the image
        h, w, _ = image.shape  # Get image dimensions
        for landmark in results.pose_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Green color, filled circle

        # Optional: Draw full landmarks with connections
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )                       

        # Display the output images
        cv2.imshow("Pose Landmarks", image)
        cv2.imshow("Pose Drawing", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No pose landmarks detected!")

# Release resources
pose.close()
