from pyzbar import pyzbar
import math
import numpy as np
import cv2


class ImageParser:

    def __init__(self):
        # self.image = cv2.imread('image-2.jpg')
        # self.image = cv2.imread('image-2.jpg')
        self.image = cv2.imread('test.png')

    def run_camera(self):
        # testing from webcam
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        while ret:
            ret, frame = camera.read()
            frame = ip.read_barcodes(frame)
            cv2.imshow('Barcode/QR code reader', frame)
            if cv2.waitKey(1) and 0xFF == 27:
                break
        camera.release()
        cv2.destroyAllWindows()

    def run(self):
        # self.find_barcodes(self.image)
        self.read_barcodes(self.image)

    @staticmethod
    def get_center(contour):
        M = cv2.moments(contour)
        cX = int(M['m10'] / max(M['m00'], 1e-6))
        cY = int(M['m01'] / max(M['m00'], 1e-6))
        return cX, cY

    @staticmethod
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def read_barcodes(image):
        file = open('barcode_result.txt', mode='a')
        barcodes = pyzbar.decode(image)
        codes = list()
        for barcode in barcodes:
            x, y, w, h = barcode.rect
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcode_data = barcode.data.decode("utf-8")
            barcode_type = barcode.type
            text = f"{barcode_data} ({barcode_type})"
            codes.append({'data': barcode_data, 'type': barcode_type})
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            print(f"[INFO] Found {barcode_type} barcode: {barcode_data}")

            file.write(f"Recognized Barcode: {barcode_data}\n")
        file.close()
        return codes

    def find_barcodes(self, image):
        """this is the final handler"""
        thresh = self.find_rectangles(image)
        contours, h = cv2.findContours(thresh, cv2.RETR_CCOMP,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(
            [contours[i] for i in range(len(contours)) if h[0][i][3] == -1],
            key=lambda x: cv2.boundingRect(x)[0])
        groups = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            center_1 = self.get_center(cnt)
            found = False
            if w * h > 50 and w * h < thresh.shape[0] * thresh.shape[1] / 2:
                for group in groups:
                    x2, y2, w2, h2 = cv2.boundingRect(group[-1])
                    center_2 = self.get_center(group[-1])
                    if abs(center_1[1] - center_2[1]) < 20 and (
                            abs(h - h2) / max(h, h2)) < 0.3 and any(
                        map(lambda p: self.dist((x, y), p) < 20,
                            [(x2, y2), (x2 + w2, y2), (x2, y2 + h2),
                             (x2 + w2, y2 + h2)])):
                        group.append(cnt)
                        found = True
                        break
                if not found:
                    groups.append([cnt])

        for group in groups[:]:
            mn = 1000000
            total = 0
            for c in group:
                x, y, w, h = cv2.boundingRect(c)
                total += w * h
                mn = min(mn, w * h)
            estimatedBars = total / mn
            if estimatedBars < 10:
                groups.remove(group)

        for idx, group in enumerate(groups):
            boxes = []
            for c in group:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append([x, y, x + w, y + h])
                cv2.rectangle(thresh, (x, y), (x + w, y + h), 255, 2)
                cv2.putText(thresh, str(idx), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
                cv2.drawContours(thresh, [c], -1, 128, -1)

            boxes = np.asarray(boxes)

            left = np.min(boxes[:, 0])
            top = np.min(boxes[:, 1])
            right = np.max(boxes[:, 2])
            bottom = np.max(boxes[:, 3])

            cv2.rectangle(image, (left, top), (right, bottom), 255, 2)

        cv2.destroyWindow("image")
        cv2.imshow("image", image)
        cv2.waitKey(0)

    def find_rectangles(self, img):
        filtered = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        for gray in cv2.split(img):
            for thrs in range(50, 200, 1):
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                contours, h = cv2.findContours(~bin, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
                contours = [contours[i] for i in range(len(contours)) if
                            h[0][i][3] == -1]
                for cnt in contours:
                    cnt_len = cv2.arcLength(cnt, True)
                    poly = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                    w, h = cv2.minAreaRect(cnt)[1]
                    if len(poly) < 8 and cv2.contourArea(
                            cnt) > 10 and cv2.contourArea(poly) < 1000 and (
                            h / w) > 5:
                        cv2.drawContours(filtered, [cnt], -1, 255, -1)
        return filtered


if __name__ == "__main__":
    ip = ImageParser()
    ip.run()
