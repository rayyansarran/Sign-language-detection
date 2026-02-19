import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Create data directory if it doesn't exist
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def collect_data():
    cap = cv2.VideoCapture(0)
    
    # Prompt for the class name (the letter or sign)
    class_name = input("Enter the name of the sign you want to collect data for: ")
    
    # Number of images to collect
    dataset_size = 100

    data = []
    
    print(f"Ready to collect data for sign: '{class_name}'. Press 'Q' to start collecting.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        cv2.putText(frame, 'Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
            
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Process frame to extract landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x, y coordinates
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data.append(data_aux)
                counter += 1
                print(f"Collected {counter}/{dataset_size} samples")

    # Save data
    file_path = os.path.join(DATA_DIR, 'data.pickle')
    
    # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            existing_data_dict = pickle.load(f)
            existing_data = existing_data_dict['data']
            existing_labels = existing_data_dict['labels']
    else:
        existing_data = []
        existing_labels = []

    existing_data.extend(data)
    existing_labels.extend([class_name] * len(data))

    with open(file_path, 'wb') as f:
        pickle.dump({'data': existing_data, 'labels': existing_labels}, f)

    print(f"Data for '{class_name}' collected and saved!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        collect_data()
        cont = input("Do you want to collect another sign? (y/n): ")
        if cont.lower() != 'y':
            break
