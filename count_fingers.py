import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

# Defina uma função para contar os dedos
def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        fingers = [0] * 5  # Inicialize o contador de dedos

        # Polegar
        if hand_landmarks.landmark[tipIds[0]].x < hand_landmarks.landmark[tipIds[0] - 1].x:
            fingers[0] = 1

        # Dedos restantes
        for i in range(1, 5):
            if hand_landmarks.landmark[tipIds[i]].y < hand_landmarks.landmark[tipIds[i] - 2].y:
                fingers[i] = 1

        total_fingers = sum(fingers)
        cv2.putText(image, str(total_fingers), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Defina uma função para desenhar os pontos de referência da mão
def drawHandLanmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

while True:
    success, image = cap.read()

    image = cv2.flip(image, 1)
    
    # Detecte os pontos de referência das mãos 
    results = hands.process(image)

    # Obtenha a posição do ponto de referência do resultado processado
    hand_landmarks = results.multi_hand_landmarks

    # Desenhe os pontos de referência
    drawHandLanmarks(image, hand_landmarks)

    # Conte os dedos
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            countFingers(image, hand_landmark)

    cv2.imshow("Controlador de Mídia", image)

    # Saia da tela ao pressionar a barra de espaço
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()
