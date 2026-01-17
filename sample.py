import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. モデルの設定 (MediaPipe Tasks)
# 手のモデルファイル (hand_landmarker.task) をダウンロードしておく必要があります
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)

# 2. カメラ映像の取得
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. 画像の処理
    # MediaPipeはRGB形式を使用するため、BGRをRGBに変換
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 4. 手の検出
    results = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

    # 5. 結果の可視化
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            for landmark in hand_landmarks:
                # 座標は正規化されているため、画面サイズを掛ける
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('MediaPipe Hand Landmarker', frame)
    if cv2.waitKey(1) & 0xFF == 27: # ESCで終了
        break

cap.release()
cv2.destroyAllWindows()
