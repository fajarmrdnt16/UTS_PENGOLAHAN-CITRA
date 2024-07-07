import streamlit as st
import cv2
import numpy as np

def upload_image():
    image_file = st.file_uploader("image", type="jpg")
    if image_file is not None:
        image_bytes = image_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    return None

def convert_rgb_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def compute_histograms(image):
    rgb_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hsv_image = convert_rgb_to_hsv(image)
    hsv_hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return rgb_hist, hsv_hist

def apply_brightness_and_contrast(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    rod = cv2.bilateralFilter(blurred, 9, 75, 75)
    lab = cv2.cvtColor(rod, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    st.title('Aplikasi Manipulasi Citra')

    image = upload_image()
    if image is not None:
        st.subheader('Gambar Asli')
        st.image(image)

        hsv_image = convert_rgb_to_hsv(image)
        st.subheader('HSV Image')
        st.image(hsv_image)

        rgb_hist, hsv_hist = compute_histograms(image)
        st.subheader('Menghitung Histogram')
        st.write(rgb_hist)

        st.subheader('Hasil Histogram')
        st.write(hsv_hist)

        final_image = apply_brightness_and_contrast(image)
        st.subheader('Gambar dengan Brightness dan Contrast')
        st.image(final_image)

        contours = find_contours(image)
        st.subheader('Contour')
        for contour in contours:
            image_with_contour = cv2.drawContours(image.copy(), [contour], 0, (0, 255, 0), 2)
            st.image(image_with_contour)

if __name__ == '__main__':
    main()
