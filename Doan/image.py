import tkinter as tk
from tkinter import filedialog,simpledialog,messagebox,ttk
from tkinter import *
import json
import cv2
import hashlib
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from keras.models import load_model
from playsound import playsound
import pygame
import subprocess
import os
import sys
import mysql.connector

def connect_db():
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="kimem1702",
            database="image_db"
        )
        if mydb.is_connected():
            print("✅ Kết nối MySQL thành công.")
    except mysql.connector.Error as e:
        print(f"❌ Lỗi kết nối MySQL: {e}")

# 🟢 Gọi hàm kiểm tra
connect_db()

# Load mô hình nhận diện biển báo
model = load_model('traffic_sign_model.h5')

labels_file = "labels.json"

classify_b = None

# Load labels from labels.json
def load_labels():
    try:
        with open(labels_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Save labels to labels.json
def save_labels():
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(classNames, f, indent=4)

classNames = load_labels()

current_image_path = None  # Lưu ảnh vừa tải lên

def get_image_hash(image): # Nếu ảnh đã có trước, lấy ra xử lý nhanh lại
    hasher = hashlib.sha256()  
    hasher.update(image.tobytes())  
    return hasher.hexdigest()  

def get_label(label):
    prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    mandatory = [33, 34, 35, 36, 37, 38, 39, 40]
    danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    if label in prohibitory:
        return "Biển báo cấm"
    elif label in mandatory:
        return "Biển báo bắt buộc"
    elif label in danger:
        return "Biển báo nguy hiểm"
    else:
        return "Biển báo khác"

# Tạo giao diện
top = tk.Tk()
top.geometry(f"1000x700+{(top.winfo_screenwidth() - 1000) // 2}+{(top.winfo_screenheight() - 700) // 2}")
img = PhotoImage(file='traffic-light.png')
top.iconphoto(False,img)
top.title('Hệ thống nhận dạng biển báo giao thông')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 25, 'bold'))
image_frame = Frame(top,borderwidth=2, relief="ridge", bg="#E0E0E0")
sign_image = Label(image_frame, bg="#E0E0E0")

def classify(file_path):
    global is_recognized
    image = Image.open(file_path) 
    image = image.resize((32, 32))
    image = np.array(image) # chuyển ảnh thành mảng numpy

    image = np.array(image).reshape(-1, 32, 32, 3)
    image = np.array(list(map(preprocessing, image)))
    image = image.reshape(-1, 32, 32, 1)
    
    Y_pred = model.predict([image])[0]
    max_prob = np.max(Y_pred)
    index = np.argmax(Y_pred)
    
    if str(index) in classNames:
        sign = classNames[str(index)]
        # Xử lý trường hợp định dạng khác nhau trong classNames
        if "\n" in sign:
            type_text, sign_name = sign.split("\n", 1)
            label.configure(foreground='#011638', text=(type_text + "\n" + sign_name))
        else:
            type_text = get_label(index)
            label.configure(foreground='#011638', text=(type_text + "\n" + sign))
        is_recognized = True  # Đánh dấu ảnh đã được nhận diện
    elif max_prob < 0.7:
        label.configure(foreground='#011638', text="Ảnh mới, chưa nhận diện")
    else:
        label.configure(foreground='#011638', text="Không xác định")
            
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img / 255

def show_classify_button(file_path):
    global classify_b
    if classify_b is not None:
        classify_b.destroy()
    classify_b = Button(top, text='Nhận diện', command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#d93303', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    global current_image_path
    try:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        current_image_path = file_path  # Lưu đường dẫn ảnh
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(current_image_path)  # Hiển thị nút nhận diện
    except:
        pass

def check_image_existence():
    global current_image_path
    if not current_image_path:
        messagebox.showerror("Lỗi", "Không có ảnh nào được chọn!")
        return False

    image = Image.open(current_image_path).resize((32, 32))
    image = np.array(image).reshape(-1, 32, 32, 3)
    image = np.array(list(map(preprocessing, image))).reshape(-1, 32, 32, 1)

    Y_pred = model.predict([image])[0]
    max_prob = np.max(Y_pred)  # Lấy xác suất cao nhất
    detected = str(np.argmax(Y_pred))  # Nhãn dự đoán
    labels = load_labels()
    if detected in labels and max_prob > 0.7:
        response = messagebox.askyesno("Thông báo","Ảnh đã có trong hệ thống, bạn chỉ có thể sửa thông tin!\nBạn có muốn sửa không?")
        if response:
            edit_label(detected, on_close=lambda: classify(current_image_path))
        else:
            show_classify_button(current_image_path)
        return True

def add_label():
    global current_image_path
    if not current_image_path:
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        current_image_path = file_path  
        
    uploaded = Image.open(current_image_path)
    uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im
    
    # Kiểm tra xem ảnh đã tồn tại trong hệ thống chưa
    image = Image.open(current_image_path).resize((32, 32))
    image = np.array(image).reshape(-1, 32, 32, 3)
    image = np.array(list(map(preprocessing, image))).reshape(-1, 32, 32, 1)

    Y_pred = model.predict([image])[0]
    max_prob = np.max(Y_pred)  # Lấy xác suất cao nhất
    detected = str(np.argmax(Y_pred))  # Nhãn dự đoán
    
    # Nếu ảnh đã có trong hệ thống với độ tin cậy cao (>= 0.7)
    if detected in classNames and max_prob >= 0.7:
        messagebox.showinfo("Thông báo", "Biển báo này đã có trong hệ thống, không thể thêm mới!")
        response = messagebox.askyesno("Thông báo", "Bạn có muốn sửa thông tin biển báo này không?")
        if response:
            edit_label(detected, on_close=lambda: classify(current_image_path))
        else:
            # Hiển thị nút nhận diện nếu người dùng không muốn sửa
            show_classify_button(current_image_path)
        return

    # Nếu ảnh chưa có trong hệ thống hoặc có độ tin cậy thấp, cho phép thêm mới
    add_window = tk.Toplevel(top)
    add_window.title("Thêm biển báo")
    add_window.geometry('200x150')
    add_window.update_idletasks()
    x = (top.winfo_screenwidth() - add_window.winfo_reqwidth()) // 2
    y = (top.winfo_screenheight() - add_window.winfo_reqheight()) // 2
    add_window.geometry(f"+{x}+{y}")
    
    tk.Label(add_window, text="Chọn loại biển báo:").pack()
    category_var = tk.StringVar()
    category_combobox = ttk.Combobox(add_window, textvariable=category_var, 
                                     values=["Biển báo cấm", "Biển báo bắt buộc", "Biển báo nguy hiểm", "Biển báo khác"])
    category_combobox.pack()
    category_combobox.current(0)
    
    tk.Label(add_window, text="Nhập tên biển báo:").pack()
    name_entry = tk.Entry(add_window)
    name_entry.pack()
    
    def save_label():
        category = category_var.get()
        name = name_entry.get().strip()
        if not name:
            messagebox.showerror("Lỗi", "Tên biển báo không được để trống!")
            return 
        
        # Lấy lại dự đoán cho ảnh hiện tại
        image = Image.open(current_image_path).resize((32, 32))
        image = np.array(image).reshape(-1, 32, 32, 3)
        image = np.array(list(map(preprocessing, image))).reshape(-1, 32, 32, 1)
        Y_pred = model.predict([image])[0]
        predicted_index = str(np.argmax(Y_pred))
        
        # Tìm ID mới cho biển báo
        available_ids = [int(k) for k in classNames.keys() if k.isdigit()]
        new_id = str(max(available_ids) + 1) if available_ids else "0"
        
        # Lưu thông tin biển báo mới
        classNames[new_id] = f"{category}\n{name}"
        save_labels()
        messagebox.showinfo("Thông báo", "Thêm thành công!")
        add_window.destroy()
        show_classify_button(current_image_path)
        
        # Cập nhật hiển thị sau khi thêm
        label_text = f"{category}\n{name}"
        label.configure(foreground='#011638', text=label_text)
        
        # Lưu ý cho người dùng về việc phải huấn luyện lại mô hình
        messagebox.showinfo("Lưu ý", "Biển báo đã được thêm vào cơ sở dữ liệu, nhưng mô hình cần được huấn luyện lại để nhận diện chính xác loại biển báo này trong tương lai.")
    
    tk.Button(add_window, text="Thêm", command=save_label, bg="green", fg="white").pack(pady=10)
        
def open_edit_label():
    global current_image_path
    if not current_image_path:
        messagebox.showerror("Lỗi", "Cần tải ảnh lên trước khi sửa")
        return

    image = Image.open(current_image_path).resize((32, 32))
    image = np.array(image).reshape(-1, 32, 32, 3)
    image = np.array(list(map(preprocessing, image))).reshape(-1, 32, 32, 1)

    Y_pred = model.predict([image])[0]
    detected_index = str(np.argmax(Y_pred))  # Lấy index dự đoán từ mô hình

    labels = load_labels()
    if detected_index in labels:
        edit_label(detected_index, on_close=lambda: classify(current_image_path))  # Cập nhật sau khi sửa
    else:
        messagebox.showerror("Lỗi", "Không tìm thấy biển báo để sửa!")
  
def reset_image():
    global current_image_path, classify_b, is_recognized, custom_labels

    # Xóa đường dẫn ảnh hiện tại và reset trạng thái nhận diện
    current_image_path = None  
    is_recognized = False  

    # Xóa ảnh hiển thị và kết quả nhận diện trên giao diện
    sign_image.configure(image='', text="")
    label.configure(text="")

    # Ẩn nút "Nhận diện" 
    if classify_b is not None:
        classify_b.destroy()
        classify_b = None

def edit_label(index,   on_close=None):
    labels = load_labels()

    if index not in labels:
        messagebox.showerror("Lỗi", "Không tìm thấy biển báo để sửa!")
        return

    label_text = labels[index]
    category, sign_name = label_text.split("\n", 1) if "\n" in label_text else ("Biển báo khác", label_text)

    edit_window = tk.Toplevel(top)
    edit_window.title("Sửa biển báo")
    edit_window.geometry('200x150')
    edit_window.update_idletasks()
    x = (top.winfo_screenwidth() - edit_window.winfo_reqwidth()) // 2
    y = (top.winfo_screenheight() - edit_window.winfo_reqheight()) // 2
    edit_window.geometry(f"+{x}+{y}")

    tk.Label(edit_window, text="Chọn loại biển báo:").pack()
    category_var = tk.StringVar(value=category)
    category_combobox = ttk.Combobox(edit_window, textvariable=category_var, 
                                     values=["Biển báo cấm", "Biển báo bắt buộc", "Biển báo nguy hiểm", "Biển báo khác"])
    category_combobox.pack()

    tk.Label(edit_window, text="Nhập tên biển báo mới:").pack()
    name_entry = tk.Entry(edit_window)
    name_entry.insert(0, sign_name)
    name_entry.pack()

    def save_edit():
        new_category = category_var.get()
        new_name = name_entry.get().strip()
        if not new_name:
            messagebox.showerror("Lỗi", "Tên biển báo không được để trống!")
            return

        labels[index] = f"{new_category}\n{new_name}"  # Lưu cả danh mục và tên biển báo
        save_labels()
        messagebox.showinfo("Thông báo", "Sửa thành công!")
        print(f"Labels sau khi sửa: {load_labels()}")
        edit_window.destroy()
        if on_close:
            on_close()
        
    tk.Button(edit_window, text="Lưu", command=save_edit, bg="blue", fg="white").pack(pady=10)

def del_img():
    pass

# Hàm để chạy webcam.py
def open_webcam():
    try:
        # Lưu trạng thái hiện tại của giao diện
        top.iconify()  # Thu nhỏ cửa sổ hiện tại
        
        # Lấy đường dẫn hiện tại của script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        webcam_path = os.path.join(current_dir, 'webcam.py')
        
        # Kiểm tra nếu file webcam.py tồn tại
        if os.path.exists(webcam_path):
            # Chạy script webcam.py trong một tiến trình riêng biệt
            if sys.platform.startswith('win'):
                subprocess.Popen(['python', webcam_path], shell=True)
            else:
                subprocess.Popen(['python3', webcam_path])
            
            # Thông báo cho người dùng
            messagebox.showinfo("Webcam", "Đang mở chế độ webcam. Nhấn 'q' để thoát khỏi chế độ webcam.")
        else:
            messagebox.showerror("Lỗi", f"Không tìm thấy file webcam.py tại {webcam_path}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể mở webcam: {str(e)}")
    finally:
        top.deiconify()  # Khôi phục cửa sổ hiện tại sau khi đóng webcam
   
frame_buttons = tk.Frame(top, bg=top["bg"])
frame_buttons.pack(side=BOTTOM,pady=50)

spacer = tk.Frame(frame_buttons, width=55, bg=top["bg"])
spacer.pack(side=RIGHT)  

del_btn = Button(frame_buttons, text="Xóa biển báo", command=del_img, padx=15, pady=8, \
                 background='#007BFF', foreground='white', font=('Arial', 10, 'bold'), \
                 relief="raised", bd=2, cursor="hand2")
del_btn.pack(side=LEFT, padx=12, pady=5)

add_btn = Button(frame_buttons, text="Thêm biển báo", command=add_label, padx=15, pady=8, \
                 background='#007BFF', foreground='white', font=('Arial', 10, 'bold'), \
                 relief="raised", bd=2, cursor="hand2")
add_btn.pack(side=LEFT, padx=12, pady=5)

upload = Button(frame_buttons, text="Tải ảnh lên", command=upload_image, padx=15, pady=8, \
                 background='#063970', foreground='white', font=('Arial', 10, 'bold'), \
                 relief="raised", bd=2, cursor="hand2")
upload.pack(side=LEFT, padx=12, pady=5)

edit_btn = Button(frame_buttons, text="Sửa biển báo", command=open_edit_label, padx=15, pady=8, \
                 background='#FFA500', foreground='white', font=('Arial', 10, 'bold'), \
                 relief="raised", bd=2, cursor="hand2")
edit_btn.pack(side=LEFT, padx=12, pady=5)

# Thêm nút webcam
webcam_btn = Button(top, text="Webcam", command=open_webcam, padx=10, pady=8,
                    background='#e91e63', foreground='white', font=('Arial', 10, 'bold'),
                    relief="raised", bd=2, cursor="hand2")

webcam_btn.place(relx=0.1, rely=0.46)


reset_btn = Button(frame_buttons, text="🔄", command=reset_image, padx=12, pady=8, \
                 background='#abb794', foreground='white', font=('Arial', 15, 'bold'), \
                 relief="raised", bd=2, cursor="hand2")  
reset_btn.pack(side=LEFT, padx=12, pady=5)

sign_image.pack(padx=5, pady=5)
image_frame.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text="Nhận dạng biển báo giao thông bằng hình ảnh", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()