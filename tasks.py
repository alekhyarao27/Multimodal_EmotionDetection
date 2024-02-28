import tkinter as tk
from tkinter import filedialog, Button, Label , Text
from PIL import Image, ImageTk
import cv2
import numpy as np
import dlib
import speech_recognition as sr
from textblob import TextBlob
import threading
from tensorflow.keras.models import load_model, model_from_json
from sklearn.preprocessing import scale

# Function to load facial expression detection model
def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Function to detect mouth openness
def detect_mouth_open(image, predictor_path, face_rect):
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face_rect)
    for i in range(48, 61):
        print("Landmark", i, ":", shape.part(i).x, shape.part(i).y)
    mouth_height = shape.part(57).y - shape.part(51).y
    face_height = face_rect.bottom() - face_rect.top()
    mouth_open_threshold = 0.15 * face_height
    print("Mouth Height:", mouth_height)
    print("Threshold:", mouth_open_threshold)
    mouth_open = mouth_height > mouth_open_threshold
    return mouth_open

# Function to detect wrinkles
def detect_wrinkles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)  # Adjust Canny edge detection parameters
    dilated_edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours (e.g., contours with area less than 100 pixels)
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    total_area = sum(cv2.contourArea(c) for c in filtered_contours)
    wrinkle_threshold = 1000  # Adjust as needed
    has_wrinkles = total_area > wrinkle_threshold
    return has_wrinkles

# Function to detect drowsiness
def detect_drowsiness(image, predictor_path, face_rect):
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face_rect)
    
    left_eye_indexes = list(range(36, 42))
    right_eye_indexes = list(range(42, 48))
    
    left_eye_ear = calculate_ear(shape, left_eye_indexes)
    right_eye_ear = calculate_ear(shape, right_eye_indexes)
    
    avg_ear = (left_eye_ear + right_eye_ear) / 2
    
    drowsiness_threshold = 0.25  # Adjust as needed
    
    is_drowsy = avg_ear < drowsiness_threshold
    return is_drowsy

def calculate_ear(shape, eye_indexes):
    left_eye_pts = [shape.part(i) for i in eye_indexes]
    left_eye_ear = eye_aspect_ratio(left_eye_pts)
    return left_eye_ear

def eye_aspect_ratio(eye_pts):
    # Calculate the distance between vertical eye landmarks
    vertical_dist1 = np.linalg.norm(np.array([eye_pts[1].x, eye_pts[1].y]) - np.array([eye_pts[5].x, eye_pts[5].y]))
    vertical_dist2 = np.linalg.norm(np.array([eye_pts[2].x, eye_pts[2].y]) - np.array([eye_pts[4].x, eye_pts[4].y]))

    # Calculate the distance between horizontal eye landmarks
    horizontal_dist = np.linalg.norm(np.array([eye_pts[0].x, eye_pts[0].y]) - np.array([eye_pts[3].x, eye_pts[3].y]))

    # Compute the eye aspect ratio
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear


# Function to upload an image file
def upload_file():
    try:
        file_path = filedialog.askopenfilename()
        file_extension = file_path.split(".")[-1]

        if file_extension.lower() in ['jpg', 'jpeg', 'png']:
            uploaded_image = Image.open(file_path)
            uploaded_image.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
            image = ImageTk.PhotoImage(uploaded_image)
            sign_image.configure(image=image)
            sign_image.image = image
            label1.configure(text='')
            show_Detect_button(file_path)  
        else:
            print("Unsupported file type")
    except Exception as e:
        print("Error:", e)

# Function to show the "Detect Emotion" button
def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=8, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.87, rely=0.5)

# Function to capture speech
def capture_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Listen for speech input
    try:
        speech_text = recognizer.recognize_google(audio)  # Recognize speech using Google Speech Recognition
        return speech_text
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
# Function to analyze emotion from speech
def analyze_emotion_from_speech(speech_text):
    speech_text_lower = speech_text.lower()
    
    # Dictionary mapping keywords to emotions
    emotion_keywords = {
        'happy': 'Happy',
        'joy': 'Happy',
        'excited': 'Happy',
        'sad': 'Sad',
        'angry': 'Angry',
        'fear': 'Fear',
        'anxious': 'Fear',
        'surprise': 'Surprise',
        'disgust': 'Disgust'
    }
    negative_keywords = [
        'unhappy', 'sad', 'angry', 'afraid', 'scared', 'terrified', 'anxious', 'worried', 'stressed',
        'depressed', 'miserable', 'gloomy', 'hopeless', 'defeated', 'disappointed', 'frustrated', 'annoyed',
        'irritated', 'enraged', 'disgusted', 'contemptuous', 'hateful', 'spiteful', 'vindictive', 'malicious',
        'not happy', 'not satisfied', 'not excited', 'feeling down', 'feeling blue', 'feeling low',
        'feeling under the weather', 'feeling on edge', 'feeling on edge', 'feeling under the weather',
        'feeling out of sorts', 'in a bad mood', 'in a foul mood', 'in a sour mood', 'in a funk', 'in a slump',
        'in a rut', 'in the dumps', 'in the doldrums', 'in the depths of despair'
    ]

    # Iterate through keywords and check if they exist in the speech text
    detected_emotion = "Neutral"  # Default emotion if no keyword is detected
    for keyword, emotion in emotion_keywords.items():
        if keyword in speech_text_lower:
            detected_emotion = emotion
            break

    # Check for negative sentiment
    sentiment = "Neutral"
    for neg_keyword in negative_keywords:
        if neg_keyword in speech_text_lower:
            sentiment = "Negative"
            break

    # Use TextBlob for sentiment analysis
    blob = TextBlob(speech_text)
    sentiment_polarity = blob.sentiment.polarity

    # Determine positive sentiment
    if sentiment_polarity > 0:
        sentiment = "Positive"
    
    print("Emotion from speech:", detected_emotion)
    print("Sentiment:", sentiment)
    #voice_output_text.insert('1.0', detected_emotion + sentiment + '\n')
    label2.configure(foreground="#011638", text="Captured Speech is "+ speech_text+"\n"+"Emotion from Speech: "+detected_emotion+",Sentiment is: "+sentiment)

# Function to detect facial emotion
def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            
            # Detect mouth openness
            mouth_open = detect_mouth_open(image, predictor_path, dlib.rectangle(x, y, x+w, y+h))
            if mouth_open:
                pred += " (Mouth Open)"
            else:
                pred += " (Mouth Closed)"

            # Detect wrinkles
            wrinkles_detected = detect_wrinkles(image)
            if wrinkles_detected:
                pred += " (Wrinkles Detected)"
            else:
                pred += " (No Wrinkles Detected)"

            # Detect drowsiness
            drowsy = detect_drowsiness(image, predictor_path, dlib.rectangle(x, y, x + w, y + h))
            if drowsy:
                pred += " (Drowsy)"
            else:
                pred += " (Alert)" 

        print("Predicted Emotion is " + pred)
        label1.configure(foreground="#011638", text=pred)
    except:
        label1.configure(foreground="#011638", text="Unable to detect")

def detect_emotion(face_image, model):
    # Preprocess the image
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    # Predict the emotion
    predicted_class = np.argmax(model.predict(face_image))
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    emotion = emotion_labels[predicted_class]

    return emotion

def start_emotion_detection():
    global stop_analysis
    stop_analysis = False  # Variable to control the loop

    cap = cv2.VideoCapture(0)

    def emotion_detection_thread():
        while not stop_analysis:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in the frame
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract the face region
                face_image = frame[y:y+h, x:x+w]

                emotion = detect_emotion(face_image, model)

                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('Emotion Detection', frame)

            # Exit if 'q' is pressed or stop_analysis is True
            if cv2.waitKey(1) & 0xFF == ord('q') or stop_analysis:
                break

        # Release the video capture object and close windows
        cap.release()
        cv2.destroyAllWindows()

    # Start the emotion detection thread
    detection_thread = threading.Thread(target=emotion_detection_thread)
    detection_thread.start()

# Function to stop real-time emotion detection
def stop_emotion_detection():
    global stop_analysis
    stop_analysis = True

# Load the facial expression detection model
model = FacialExpressionModel("model_a1.json", "model_weights1.h5")

# Path to the facial landmark predictor file
predictor_path = "shape_predictor_68_face_landmarks.dat"

# List of emotion labels
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label2 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
upload = Button(top, text="Upload Image", command=upload_file, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
upload.place(relx=0.54, rely=0.95, anchor='sw')
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
label2.pack(side='bottom', expand='True')
heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

speech_button = Button(top, text="Capture Speech", command=lambda: analyze_emotion_from_speech(capture_speech()))
speech_button.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
speech_button.place(relx=0.77, rely=0.95, anchor='sw')

start_analysis_button = Button(top, text="Start Video Analysis", command=start_emotion_detection)
start_analysis_button.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
start_analysis_button.place(relx=0.01, rely=0.95, anchor='sw')

stop_analysis_button = Button(top, text="Stop Video Analysis", command=stop_emotion_detection)
stop_analysis_button.configure(background="#364156", foreground='white', font=('arial', 15, 'bold'))
stop_analysis_button.place(relx=0.27, rely=0.95, anchor='sw')
top.mainloop()
