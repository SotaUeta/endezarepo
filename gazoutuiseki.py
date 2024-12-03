#解説1
import cv2
import numpy as np

#解説2
# ShiTomasiのコーナー検出のためのパラメータ
fparams = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# lucas kanadeのオプティカルフローのためのパラメータ
lparams = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#解説3
# ランダムな色を生成
color = np.random.randint(0,255,(100,3))

# ビデオを読み込む
cap = cv2.VideoCapture(0)

# 最初のフレームを取得し、それに含まれるコーナーを見つける
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **fparams)

# 描画用のマスク画像を作成
mask = np.zeros_like(old_frame)

#解説4
while(1):
    ret,frame = cap.read()
    if not ret or frame is None:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # オプティカルフローを計算
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lparams)

    # 良い点を選択
    good_new = p1[st==1]
    good_old = p0[st==1]

    # トラックを描画
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        a,b,c,d = int(a), int(b), int(c), int(d)  # 座標を整数に変換
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    # 画像を表示
    cv2.imshow('Optical Flow', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 前のフレームと前の点を更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()