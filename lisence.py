import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageFilter


img = cv2.imread('/Users/apple/Desktop/python_lesson/asstes/license_0.jpg')
# img = cv2.imread('/Users/apple/Desktop/python_lesson/asstes/license_1.png')
# img = cv2.imread('/Users/apple/Desktop/python_lesson/asstes/license_2.jpg')
# img = cv2.imread('/Users/apple/Desktop/python_lesson/asstes/lisence_3.jpg')
# 画像の前処理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 200)

# 輪郭検出
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print("contours:", len(contours))
# 最大の四角形を取得
for c in contours:
    peri = cv2.arcLength(c, True)
    epsilon = 0.01 * peri  # 近似多角形の精度を調整
    approx = cv2.approxPolyDP(c, epsilon, True) # 近似された多角形の各頂点の座標
    print("approxの数:", len(approx))
    if len(approx) == 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_img = img[y:y + h, x:x + w]  # 最大の四角形を切り抜く

        # 切り抜いた領域の比率に基づいて座標を計算
        # region_ratio = (0.375, 0.745, 0.435, 0.865)
        region_ratio = (0.35, 0.73, 0.65, 0.98)
        region_x1 = int(x + region_ratio[0] * w)
        region_y1 = int(y + region_ratio[1] * h)
        region_x2 = int(x + region_ratio[2] * w)
        region_y2 = int(y + region_ratio[3] * h)
        

        # 四角形に切り抜いた領域を比率で切り抜く
        cropped_region = cropped_img[region_y1:region_y2, region_x1:region_x2]
        
        # 切り抜いた領域からさらに最大の四角形を見つけて切り抜く
        cropped_gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
        cropped_blur = cv2.GaussianBlur(cropped_gray, (5, 5), 0)
        cropped_edges = cv2.Canny(cropped_blur, 30, 100)

        # # ノイズ除去のためのフィルタリング処理を適用
        # kernel = np.ones((3, 3), np.uint8)
        # cropped_edges = cv2.morphologyEx(cropped_edges, cv2.MORPH_OPEN, kernel)

        cropped_contours, _ = cv2.findContours(cropped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cropped_contours = sorted(cropped_contours, key=cv2.contourArea, reverse=True)

        print("cropped_contours:", len(cropped_contours))
        for c in cropped_contours:
            peri = cv2.arcLength(c, True)
            epsilon = 0.01 * peri
            approx = cv2.approxPolyDP(c, epsilon, True)
            # print("approxの数2:", len(approx))
            # if len(approx) == 4:
            if len(approx) == 4 and cv2.isContourConvex(approx):
                print("四角形です")
                print("approx:", approx)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(cropped_region, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cropped_max_rect = cropped_region[y:y + h, x:x + w]  # 最大の四角形を切り抜く

                # 文字検出
                smoothed_img_pil = Image.fromarray(cropped_max_rect)
                gray_img = smoothed_img_pil.convert('L')  # グレースケールに変換
                config = '-l jpn --oem 3 --psm 6'
                text = pytesseract.image_to_string(gray_img, config=config)
                print("region:", (region_x1, region_y1, region_x2, region_y2))
                print("approx:", approx)
                print("文字:", text)
                # 結果の表示
                cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
                cv2.imshow('Result', cropped_region)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    # else:
        # print("条件を満たさない輪郭:", approx)