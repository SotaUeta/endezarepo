# coding: UTF-8
import cv2
import serial
import datetime

#トラックバー処理
def onTrackbarT(position):
    global threshold
    threshold = position

def onTrackbarH(position):
    global hour
    hour = position

def onTrackbarM(position):
    global min
    min = position

def onTrackbarS(position):
    global sec
    sec = position

#サーボ駆動
def servo_move(ser, pos1, pos2):
    str1 = str(pos1)
    str2 = str(pos2)

    ser.write(str1.zfill(5)+str2.zfill(5))

#メイン関数
def main():
    #パラメータ初期化
    global threshold, hour, min, sec
    threshold = 232
    hour, min, sec = 15, 50, 0

    #サーボ移動量(ズーム無しでは10、ズーム有りでは5に)
    step = 5

    try:
        ser = serial.Serial(6, 115200) #COM7    
    except:
        print(u"シリアルポートがオープンできません")
        exit()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("View")
    cv2.createTrackbar("Threshold", "View", threshold, 255, onTrackbarT)
    cv2.createTrackbar("hour", "View", hour, 23, onTrackbarH)
    cv2.createTrackbar("min", "View", min, 59, onTrackbarM)
    cv2.createTrackbar("sec", "View", sec, 59, onTrackbarS)

    #サーボ初期位置
    posx = 7500
    posy = 10167
    servo_move(ser, posx, posy)

    while(True):
        ret, img = cap.read()
        #グレースケール
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ガウスフィルタ
        img_g = cv2.GaussianBlur(img_g, (15,15), 0)
        #2値化
        ret, img_g = cv2.threshold(img_g, threshold, 255, cv2.THRESH_BINARY)
        #カラー化
        img_out = cv2.cvtColor(img_g, cv2.COLOR_GRAY2BGR)

        #輪郭抽出
        contours, hierarchy = cv2.findContours(img_g, 1, 2)

        #矩形を求める
        mw, mh = 0, 0
        flg_detect = 0
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 1)
            if mw*mh < w*h:
                mw, mh, mx, my = w, h, x, y
                flg_detect = 1

        #最大面積の矩形を赤で
        if flg_detect == 1:
            cv2.rectangle(img_out, (mx, my), (mx+mw, my+mh), (0, 50, 255), 1)

        #現在時刻と起動時刻の比較
        t_start = datetime.time(hour, min, sec)
        t_lock = datetime.time(hour, min, sec+2)
        t_up = datetime.time(hour, min, sec+10)
        t_now = datetime.datetime.now()

        if t_start > t_now.time():
            #待機
            print("wait")
        elif t_lock > t_now.time():
            #3秒までは固定
            print("lock")
        elif t_up > t_now.time():
            #10秒までは上昇のみ
            print("go up")
            if flg_detect == 1:
                if my < 240:
                    posy -= step

                servo_move(ser, posx, posy)
        else:
            #10秒以降は上昇+右
            print("go up and right")
            if flg_detect == 1:
                if mx+mw > 320:
                    posx += step

                if my < 240:
                    posy -= step

                servo_move(ser, posx, posy)

        #ウィンドウ表示
        img_view = cv2.hconcat([img, img_out])
        cv2.imshow("View",img_view)

        if cv2.waitKey(10) > 0:
            break

    #終了処理
    cap.release()
    cv2.destroyAllWindows()

#メイン関数実行
main()