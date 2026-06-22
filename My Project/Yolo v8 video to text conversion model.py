import cv2
import sounddevice as sd
import numpy as np
import os
import time
import smtplib
from email.message import EmailMessage
from gtts import gTTS
from ultralytics import YOLO

# Email configuration
SENDER_EMAIL = "meenassk17@gmail.com"  # Replace with your email
PASSWORD = "mavl isxe wbzm wsmz"  # Replace with your email password or app password
RECEIVER_EMAIL = "meenassk17@gmail.com"  # Replace with recipient's email

# Function to send an email alert
def send_email_alert():
    msg = EmailMessage()
    msg["Subject"] = "⚠️ Alert: Person Detected!"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content("A person has been detected by the camera.")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, PASSWORD)
            server.send_message(msg)
        print("📩 Email alert sent successfully!")
    except Exception as e:
        print(f"❌ Email failed to send: {e}")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Couldn't access the webcam.")
    exit()

# Sound settings
sample_rate = 44100  # CD-quality audio
freq = 1000  # Frequency of buzzer sound (Hz)
duration = 0.5  # Buzzer duration (seconds)

# Generate a buzzer sound wave
t = np.linspace(0, duration, int(sample_rate * duration), False)
wave = 0.5 * np.sin(2 * np.pi * freq * t)

# Generate voice alert
alert_file = "alert.mp3"
tts = gTTS("A person is detected.", lang="en")
tts.save(alert_file)

def play_buzzer():
    """Plays a buzzer sound."""
    sd.play(wave, samplerate=sample_rate)
    sd.wait()

def speak_alert():
    """Speaks 'A person is detected'."""
    os.system(f"start {alert_file}")

person_detected = False
email_sent = False  # Track if email was sent

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Check if a "person" is detected
    detected = False
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Get class index
            label = model.names[cls]  # Get class name
            if label == "person":
                detected = True
                break

    # If a person is detected, play buzzer and alert in sequence
    if detected:
        if not person_detected:  # First detection
            print("A person is detected.")
            if not email_sent:  # Only send email once per detection
                send_email_alert()
                email_sent = True  # Avoid spamming emails

        play_buzzer()  # Step 1: Play buzzer
        speak_alert()  # Step 2: Speak alert
        time.sleep(0.9)

        person_detected = True

    else:
        person_detected = False
        email_sent = False  # Reset email flag when no person is found

    # Display detection results
    for r in results:
        frame = r.plot()

    cv2.imshow("YOLOv8 Detection with Voice Alert & Buzzer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
