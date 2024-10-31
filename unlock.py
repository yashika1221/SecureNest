import cv2
import os
import numpy as np
#libraries for email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.mime.image import MIMEImage

# function for email 
def send_email(sender_email, receiver_email, subject, body, password, image_path=None):
    try:
        # Create the email message object
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # Attach the body text
        msg.attach(MIMEText(body, 'plain'))

        # If an image path is provided, attach the image
        if image_path:
            with open(image_path, 'rb') as img_file:
                img = MIMEImage(img_file.read())
                img.add_header('Content-Disposition', f'attachment; filename="{image_path}"')
                msg.attach(img)

        # Connect to Gmail's SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Use TLS encryption

        # Log in to the server
        server.login(sender_email, password)

        # Send the email
        server.sendmail(sender_email, receiver_email, msg.as_string())

        # Close the connection
        server.quit()

        print("Email sent successfully!")

    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# Email details
sender_email = "iotp256@gmail.com"
receiver_email = "123102035@nitkkr.ac.in"
subject = "Unidentified Person"
body = "Knock Knock!! \nThere's an unidentified person at the door."
password = "mrod rebz qmjb gwzh"  # Or use an app-specific password
image_path = "captured.jpg"  # Provide the path to your image file
# Function to capture an image from the camera
def capture_image():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()
    if ret:
        image_filename = os.path.join(os.getcwd(), "captured.jpg")
        cv2.imwrite(image_filename, frame)  # Save the captured image
        cap.release()
        cv2.destroyAllWindows()
        return image_filename
    else:
        print("Error: Could not read frame.")
        cap.release()
        cv2.destroyAllWindows()
        return None

# Function to train the face recognizer with user photos
def train_model(user_photos_folder, img_size=(100, 100)):
    # Create the face recognizer using LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    # Load photos from each user's folder
    for user_folder in os.listdir(user_photos_folder):
        user_path = os.path.join(user_photos_folder, user_folder)

        if os.path.isdir(user_path):
            label_map[current_label] = user_folder
            for img_file in os.listdir(user_path):
                img_path = os.path.join(user_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    # Resize image to fixed size
                    img_resized = cv2.resize(img, img_size)
                    faces.append(img_resized)
                    labels.append(current_label)

            current_label += 1

    faces = np.array(faces)
    labels = np.array(labels)

    # Train the model
    face_recognizer.train(faces, labels)
    return face_recognizer, label_map

# Function to unlock the gate (simulated)
def unlock_gate(user):
    print(f"Gate Unlocked for {user}!")

# Main program logic
if _name_ == "_main_":
    user_photos_folder = 'photos'  # Folder with user photos
    img_size = (300, 300)  # Fixed size for all images (can adjust based on dataset)
    
    # Train model with images resized to 100x100
    model, label_map = train_model(user_photos_folder, img_size)

    # Capture an image at the gate
    captured_image_path = capture_image()
    if captured_image_path is None:
        exit()

    # Read the captured image in grayscale
    captured_image = cv2.imread(captured_image_path, cv2.IMREAD_GRAYSCALE)

    if captured_image is not None:
        # Detect faces in the captured image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces_detected = face_cascade.detectMultiScale(captured_image, scaleFactor=1.1, minNeighbors=5)

        if len(faces_detected) == 0:
            print("No face detected.")
            exit()

        # We assume the first face found is the target for simplicity
        (x, y, w, h) = faces_detected[0]
        face_roi = captured_image[y:y+h, x:x+w]

        # Resize the face region to the same size as used in training
        face_roi_resized = cv2.resize(face_roi, img_size)

        # Recognize the face using the trained model
        label, confidence = model.predict(face_roi_resized)

        # Confidence threshold to determine if it's a match (lower is better)
        if confidence < 45:  # You can adjust the threshold based on your dataset and needs
            recognized_user = label_map[label]
            print(f"Face recognized! User: {recognized_user}, Confidence: {confidence}")
            unlock_gate(recognized_user)
        else:
            print(f"Face not recognized. Confidence: {confidence}")
            send_email(sender_email, receiver_email, subject, body, password, image_path)

    else:
        print("Error: Could not load the captured image.")
