import cv2
import easyocr
import numpy as np
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

# Inisialisasi tema CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Variabel global
image_original = None
image_display = None
processed_image = None
processing_mode = None
cap = None
is_camera_active = False

# Fungsi untuk resize gambar
def resize_image(image, max_width=1000, max_height=800):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    if scaling_factor < 1:
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return cv2.resize(image, (new_width, new_height))
    return image

# Fungsi untuk update preview kamera
def update_camera_preview():
    global cap, is_camera_active
    if is_camera_active and cap is not None:
        ret, frame = cap.read()
        if ret:
            frame = resize_image(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            image_label.configure(image=image_tk)
            image_label.image = image_tk
            if is_camera_active:  # Check if camera is still active
                root.after(10, update_camera_preview)  # Schedule next update

# Fungsi untuk menangkap gambar dari kamera
def capture_image():
    global image_original, image_display, processed_image, cap, is_camera_active
    if cap is not None and is_camera_active:
        ret, frame = cap.read()
        if ret:
            image_original = resize_image(frame)
            image_display = image_original.copy()
            processed_image = image_original.copy()
            update_image(image_display)
            process_image()
            
            # Matikan kamera dan sembunyikan tombol shutter
            stop_camera()
            shutter_button.pack_forget()
        else:
            print("Error: Gagal menangkap gambar.")

# Fungsi untuk memulai kamera
def start_camera():
    global cap, is_camera_active
    cap = cv2.VideoCapture(0)
    # Set resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)  # Atur lebar frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)  # Atur tinggi frame
    if cap.isOpened():
        is_camera_active = True
        shutter_button.pack(pady=10)  # Tampilkan tombol shutter
        update_camera_preview()
    else:
        print("Error: Kamera tidak tersedia.")

# Fungsi untuk menghentikan kamera
def stop_camera():
    global cap, is_camera_active
    is_camera_active = False
    if cap is not None:
        cap.release()
        cap = None

# Fungsi untuk memuat gambar atau menangkap gambar dari kamera
def load_image_with_option():
    def on_option_selected(choice):
        global image_original, image_display, processed_image
        if choice == "Load Image":
            stop_camera()  # Pastikan kamera mati
            shutter_button.pack_forget()  # Sembunyikan tombol shutter
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
            if file_path:
                image_original = cv2.imread(file_path)
                image_original = resize_image(image_original)
                image_display = image_original.copy()
                processed_image = image_original.copy()
                update_image(image_display)
                process_image()
        elif choice == "Capture Image":
            start_camera()

    # Menu pilihan untuk metode input
    option_menu = ctk.CTkOptionMenu(control_frame, values=["Load Image", "Capture Image"], command=on_option_selected)
    option_menu.pack(pady=10)

# Fungsi untuk memperbarui gambar di GUI
def update_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)
    image_label.configure(image=image_tk)
    image_label.image = image_tk

# Fungsi untuk pra-pemrosesan
def process_image(event=None):
    global image_display, processed_image, processing_mode
    if image_original is not None:
        if processing_mode == "Grayscale":
            processed_image = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
            image_display = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

        elif processing_mode == "Binary":
            gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
            kernel_size = int(2 * gauss_slider.get() + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            _, processed_image = cv2.threshold(blurred, binary_thresh_slider.get(), 255, cv2.THRESH_BINARY)
            image_display = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

        elif processing_mode == "Canny":
            gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
            kernel_size = int(2 * gauss_slider.get() + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            processed_image = cv2.Canny(blurred, canny_slider1.get(), canny_slider2.get())
            image_display = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

        elif processing_mode == "Sobel":  
            gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
            kernel_size = int(2 * gauss_slider.get() + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
            
            # Aplikasikan filter Sobel berdasarkan arah yang dipilih
            sobel_direction = sobel_direction_var.get()
            if sobel_direction == "X":
                sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            elif sobel_direction == "Y":
                sobel = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            else:  # XY
                sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalisasi hasil Sobel
            sobel_abs = np.abs(sobel)
            sobel_normalized = cv2.normalize(sobel_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Aplikasikan threshold
            _, processed_image = cv2.threshold(sobel_normalized, sobel_thresh_slider.get(), 255, cv2.THRESH_BINARY)
            image_display = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        update_image(image_display)

# Fungsi untuk menjalankan OCR
def run_ocr():
    global processed_image
    if processed_image is not None:
        reader = easyocr.Reader(['en'])
        result = reader.readtext(processed_image)
        text_output.configure(state="normal")
        text_output.delete("1.0", "end")
        confidences = []
        detected_texts = []

        for (bbox, text, confidence) in result:
            (top_left, bottom_right) = bbox[0], bbox[2]
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            confidences.append(confidence)
            detected_texts.append(text)
            cv2.rectangle(image_display, top_left, bottom_right, (0, 255, 0), 2)
            text_position = (top_left[0], top_left[1] - 10)
            cv2.putText(image_display, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            text_output.insert("end", f"Text: {text}\nConfidence: {confidence:.2f}\n")

        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            text_output.insert("end", f"\nAverage Confidence: {avg_confidence:.2f}\n")
        
        if detected_texts:
            all_texts = " ".join(detected_texts)
            text_output.insert("end", f"\n{all_texts}")

        text_output.configure(state="disabled")
        update_image(image_display)

# Fungsi untuk menyimpan gambar hasil deteksi
def save_result():
    global image_display
    if image_display is not None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, image_display)
            print(f"Gambar hasil deteksi berhasil disimpan di: {file_path}")

# Fungsi untuk mereset gambar
def reset_image():
    global image_display, processed_image
    if image_original is not None:
        processed_image = image_original.copy()
        image_display = image_original.copy()
        update_image(image_display)
        text_output.configure(state="normal")
        text_output.delete("1.0", "end")
        text_output.configure(state="disabled")

# Fungsi untuk memperbarui mode
def update_mode(new_mode):
    global processing_mode
    processing_mode = new_mode
    binary_thresh_slider_frame.pack_forget()
    canny_slider1_frame.pack_forget()
    canny_slider2_frame.pack_forget()
    sobel_direction_frame.pack_forget()
    sobel_thresh_frame.pack_forget()
    gauss_slider_frame.pack_forget() 

    if processing_mode == "Binary":
        binary_thresh_slider_frame.pack(pady=5)
        gauss_slider_frame.pack(pady=5)
    elif processing_mode == "Canny":
        canny_slider1_frame.pack(pady=5)
        canny_slider2_frame.pack(pady=5)
        gauss_slider_frame.pack(pady=5)
    elif processing_mode == "Sobel":  
        sobel_direction_frame.pack(pady=5)
        sobel_thresh_frame.pack(pady=5)
        gauss_slider_frame.pack(pady=5)
    process_image()

# Fungsi cleanup saat aplikasi ditutup
def on_closing():
    stop_camera()
    root.destroy()

# GUI setup dengan CustomTkinter
root = ctk.CTk()
root.title("Interactive Text Detection")
root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle window closing

# Frame kontrol
control_frame = ctk.CTkFrame(root)
control_frame.pack(side="left", fill="y", padx=10, pady=10)

load_image_with_option()

# Tombol shutter (awalnya tersembunyi)
shutter_button = ctk.CTkButton(control_frame, text="Capture", command=capture_image)

mode_var = ctk.StringVar(value="Grayscale")
mode_menu = ctk.CTkOptionMenu(control_frame, 
                             variable=mode_var, 
                             values=["Grayscale", "Binary", "Canny", "Sobel"],
                             command=update_mode)
mode_menu.pack(pady=10)

binary_thresh_slider_frame = ctk.CTkFrame(control_frame)
binary_thresh_slider_label = ctk.CTkLabel(binary_thresh_slider_frame, text="Binary Threshold")
binary_thresh_slider_label.pack()
binary_thresh_slider = ctk.CTkSlider(binary_thresh_slider_frame, from_=0, to=255, command=process_image)
binary_thresh_slider.set(127)
binary_thresh_slider.pack()

canny_slider1_frame = ctk.CTkFrame(control_frame)
canny_slider1_label = ctk.CTkLabel(canny_slider1_frame, text="Canny Threshold 1")
canny_slider1_label.pack()
canny_slider1 = ctk.CTkSlider(canny_slider1_frame, from_=0, to=255, command=process_image)
canny_slider1.set(50)
canny_slider1.pack()

canny_slider2_frame = ctk.CTkFrame(control_frame)
canny_slider2_label = ctk.CTkLabel(canny_slider2_frame, text="Canny Threshold 2")
canny_slider2_label.pack()
canny_slider2 = ctk.CTkSlider(canny_slider2_frame, from_=0, to=255, command=process_image)
canny_slider2.set(150)
canny_slider2.pack()

gauss_slider_frame = ctk.CTkFrame(control_frame)
gauss_slider_label = ctk.CTkLabel(gauss_slider_frame, text="Gaussian Blur")
gauss_slider_label.pack()
gauss_slider = ctk.CTkSlider(gauss_slider_frame, from_=0, to=20, command=process_image)
gauss_slider.set(0)
gauss_slider.pack()

sobel_direction_frame = ctk.CTkFrame(control_frame)
sobel_direction_label = ctk.CTkLabel(sobel_direction_frame, text="Sobel Direction")
sobel_direction_label.pack()
sobel_direction_var = ctk.StringVar(value="XY")
sobel_direction_menu = ctk.CTkOptionMenu(sobel_direction_frame, 
                                        variable=sobel_direction_var,
                                        values=["X", "Y", "XY"],
                                        command=lambda x: process_image())
sobel_direction_menu.pack()
sobel_thresh_frame = ctk.CTkFrame(control_frame)
sobel_thresh_label = ctk.CTkLabel(sobel_thresh_frame, text="Sobel Threshold")
sobel_thresh_label.pack()
sobel_thresh_slider = ctk.CTkSlider(sobel_thresh_frame, from_=0, to=255, command=process_image)
sobel_thresh_slider.set(127)
sobel_thresh_slider.pack()

ocr_button = ctk.CTkButton(control_frame, text="Run OCR", command=run_ocr)
ocr_button.pack(pady=10)

reset_button = ctk.CTkButton(control_frame, text="Reset Image", command=reset_image)
reset_button.pack(pady=10)

save_button = ctk.CTkButton(control_frame, text="Save Image", command=save_result)
save_button.pack(pady=10)

output_frame = ctk.CTkFrame(root)
output_frame.pack(side="left", fill="y", padx=10, pady=10)

text_output = ctk.CTkTextbox(output_frame, width=300, height=600)
text_output.pack(pady=10)
text_output.configure(state="disabled")

image_label = ctk.CTkLabel(root, text="", anchor="center")
image_label.pack(side="right", padx=20, pady=20, expand=True, fill="both")

root.mainloop()
