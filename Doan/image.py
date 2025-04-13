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
import subprocess
import os
import sys
import mysql.connector
from mysql.connector import Error
import threading
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

def preprocess_for_training(img):
    """Tiền xử lý ảnh cho quá trình huấn luyện"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img / 255

def connect_db():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="kimem1702",
        database="image_db"
    )
    return mydb
def get_labels_from_db():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="kimem1702",
        database="image_db"
    )
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT id, class_id, class_name FROM labels")
    labels = cursor.fetchall()
    cursor.close()
    connection.close()
    
    # Chuyển đổi kết quả thành dictionary với class_id là key
    labels_dict = {}
    for label in labels:
        labels_dict[str(label['class_id'])] = label['class_name']
    return labels_dict

def fine_tune_model(new_images, new_labels, model_path='traffic_sign_model.h5', epochs=5):
    """Fine-tune mô hình hiện có với dữ liệu mới"""
    from keras.models import load_model
    import numpy as np
    
    # Lấy số lượng lớp từ cơ sở dữ liệu
    connection = connect_db()
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(DISTINCT class_id) as num_classes FROM labels")
    num_classes = cursor.fetchone()[0]
    cursor.close()
    connection.close()
    
    # Load mô hình đã được huấn luyện
    model = load_model(model_path)
    
    # Chuẩn bị dữ liệu mới
    processed_images = []
    for img in new_images:
        img = preprocess_for_training(img)
        processed_images.append(img)
    
    new_images = np.array(processed_images)
    new_images = new_images.reshape(-1, 32, 32, 1)
    new_labels = to_categorical(new_labels, num_classes)
    
    # Đặt tỷ lệ học thấp hơn cho fine-tuning
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    # Fine-tune mô hình với dữ liệu mới
    history = model.fit(new_images, new_labels, 
                      batch_size=16, 
                      epochs=epochs, 
                      validation_split=0.2,
                      shuffle=True)
    
    # Lưu mô hình đã fine-tune
    model.save(model_path)
    return model

def update_model_from_db():
    """Cập nhật mô hình với dữ liệu mới từ database"""
    try:
        connection = connect_db()
        cursor = connection.cursor(dictionary=True)
        
        # Đảm bảo có cột trained trong bảng images
        try:
            cursor.execute("SHOW COLUMNS FROM images LIKE 'trained'")
            result = cursor.fetchone()
            if not result:
                cursor.execute("ALTER TABLE images ADD COLUMN trained TINYINT DEFAULT 0")
                connection.commit()
        except mysql.connector.Error:
            # Tạo cột trained nếu chưa tồn tại
            cursor.execute("ALTER TABLE images ADD COLUMN trained TINYINT DEFAULT 0")
            connection.commit()
        
        # Lấy các ảnh đã thêm chưa được sử dụng để huấn luyện
        cursor.execute("SELECT images.path, labels.class_id FROM images JOIN labels ON images.label_id = labels.id WHERE images.trained = 0")
        new_data = cursor.fetchall()
        
        if not new_data:
            messagebox.showinfo("Thông báo", "Không có dữ liệu mới để cập nhật mô hình!")
            return False
            
        # Chuẩn bị dữ liệu
        new_images = []
        new_labels = []
        
        for item in new_data:
            img = cv2.imread(item['path'])
            if img is None:
                messagebox.showerror("Lỗi", f"Không thể đọc ảnh từ đường dẫn: {item['path']}")
                continue
                
            img = cv2.resize(img, (32, 32))
            new_images.append(img)
            new_labels.append(item['class_id'])
        
        if not new_images:
            messagebox.showinfo("Thông báo", "Không có ảnh hợp lệ để cập nhật mô hình!")
            return False
            
        # Cập nhật mô hình
        global model
        model = fine_tune_model(new_images, new_labels, epochs=3)       
        
        # Đánh dấu các ảnh đã được huấn luyện
        cursor.execute("UPDATE images SET trained = 1 WHERE trained = 0")
        connection.commit()
        cursor.close()
        connection.close()
        
        return True
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể cập nhật mô hình: {str(e)}")
        return False

def update_model():
    """Hàm cập nhật mô hình khi nhấn nút"""
    # Tạo cửa sổ progress
    progress_window = tk.Toplevel(top)
    progress_window.title("Đang cập nhật mô hình...")
    progress_window.geometry("300x100")
    progress_window.resizable(False, False)
    x = (top.winfo_screenwidth() - 300) // 2
    y = (top.winfo_screenheight() - 100) // 2
    progress_window.geometry(f"+{x}+{y}")
    
    tk.Label(progress_window, text="Đang cập nhật mô hình, vui lòng đợi...").pack(pady=10)
    progress_bar = ttk.Progressbar(progress_window, orient="horizontal", mode="indeterminate")
    progress_bar.pack(fill=tk.X, padx=20)
    progress_bar.start()
    
    # Thực hiện cập nhật trong một luồng riêng biệt
    def update_thread():
        success = update_model_from_db()
        progress_window.destroy()
        if success:
            messagebox.showinfo("Thành công", "Mô hình đã được cập nhật thành công!")
            # Cập nhật lại danh sách nhãn
            global labels, classNames
            labels = get_labels_from_db()
            classNames = labels
        
    thread = threading.Thread(target=update_thread)
    thread.daemon = True
    thread.start()

global labels
# Khởi tạo labels
labels = get_labels_from_db()
classNames = labels  # classNames giờ là dictionary với key là class_id
# Load mô hình nhận diện biển báo
model = load_model('traffic_sign_model.h5')
classify_b = None
current_image_path = None  # Lưu ảnh vừa tải lên
is_recognized = False      # Trạng thái nhận diện ảnh

def preprocess_image(file_path):
    """Tiền xử lý ảnh từ đường dẫn file"""
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image = np.array(image).reshape(-1, 32, 32, 3)
    image = np.array(list(map(preprocessing, image)))
    image = image.reshape(-1, 32, 32, 1)
    return image

def predict_label(processed_image):
    """Dự đoán nhãn từ ảnh đã xử lý"""
    Y_pred = model.predict([processed_image])[0]
    max_prob = np.max(Y_pred)
    predicted_index = np.argmax(Y_pred)
    return str(predicted_index), max_prob

def insert_label(class_id, class_name):
    try:
        connection = connect_db()
        cursor = connection.cursor()
        query = "INSERT INTO labels (class_id, class_name) VALUES (%s, %s)"
        cursor.execute(query, (class_id, class_name))
        connection.commit()
        label_id = cursor.lastrowid  # Lấy ID tự tăng của label vừa thêm
        cursor.close()
        connection.close()
        return label_id
    except mysql.connector.Error as e:
        messagebox.showerror("Lỗi", f"Không thể thêm label: {e}")
        return None

def insert_image(title, description, path, label_id):
    try:
        connection = connect_db()
        cursor = connection.cursor()
        query = "INSERT INTO images (title, description, path, label_id) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (title, description, path, label_id))
        connection.commit()
        cursor.close()
        connection.close()
    except mysql.connector.Error as e:
        messagebox.showerror("Lỗi", f"Không thể thêm hình ảnh: {e}")


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
    processed_image = preprocess_image(file_path)
    predicted_index, max_prob = predict_label(processed_image)
    
    if max_prob >= 0.7:  # Chỉ nhận diện khi độ tin cậy đủ cao
        if predicted_index in classNames:
            sign = classNames[predicted_index]
            # Xử lý trường hợp định dạng khác nhau trong classNames
            if "\n" in sign:
                type_text, sign_name = sign.split("\n", 1)
                label.configure(foreground='#011638', text=(type_text + "\n" + sign_name))
            else:
                type_text = get_label(int(predicted_index))
                label.configure(foreground='#011638', text=(type_text + "\n" + sign))
            is_recognized = True  # Đánh dấu ảnh đã được nhận diện
        else:
            label.configure(foreground='#011638', text="Biển báo chưa được đăng ký")
    else:
        label.configure(foreground='#011638', text="Độ tin cậy thấp (< 70%)")

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

    processed_image = preprocess_image(current_image_path)
    detected_index, max_prob = predict_label(processed_image)
    
    if detected_index in labels and max_prob > 0.7:
        response = messagebox.askyesno("Thông báo","Ảnh đã có trong hệ thống, bạn chỉ có thể sửa thông tin!\nBạn có muốn sửa không?")
        if response:
            edit_label(detected_index, on_close=lambda: classify(current_image_path))
        else:
            show_classify_button(current_image_path)
        return True
    return False

def add_label():
    global current_image_path, labels
    
    # Biến để xác định nguồn ảnh (ảnh hiện tại hoặc ảnh mới)
    selected_image_path = None
    
    # Nếu đã có ảnh, hỏi người dùng có muốn dùng ảnh hiện tại hay không
    if current_image_path:
        response = messagebox.askyesno("Thông báo", 
                                      "Bạn có muốn dùng ảnh hiện tại để thêm biển báo không?\n(Chọn 'No' để tải ảnh mới)")
        if response:
            selected_image_path = current_image_path
        else:
            file_path = filedialog.askopenfilename()
            if not file_path:
                return  # Người dùng đã hủy việc chọn file
            selected_image_path = file_path
            # Cập nhật current_image_path và hiển thị ảnh mới
            current_image_path = selected_image_path
    else:
        # Nếu chưa có ảnh, mở hộp thoại chọn file
        file_path = filedialog.askopenfilename()
        if not file_path:
            return  # Người dùng đã hủy việc chọn file
        selected_image_path = file_path
        current_image_path = selected_image_path
    
    # Hiển thị ảnh đã chọn
    uploaded = Image.open(selected_image_path)
    uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
    im = ImageTk.PhotoImage(uploaded)
    sign_image.configure(image=im)
    sign_image.image = im
    
    # Kiểm tra xem ảnh đã tồn tại trong hệ thống chưa
    processed_image = preprocess_image(selected_image_path)
    predicted_index, max_prob = predict_label(processed_image)
    
    if predicted_index in labels and max_prob >= 0.7:
        messagebox.showinfo("Thông báo", "Biển báo này đã có trong hệ thống, không thể thêm mới!")
        response = messagebox.askyesno("Thông báo", "Bạn có muốn sửa thông tin biển báo này không?")
        if response:
            edit_label(predicted_index, on_close=lambda: classify(selected_image_path))
        else:
            show_classify_button(selected_image_path)
        return

    # Nếu ảnh chưa có trong hệ thống hoặc có độ tin cậy thấp, cho phép thêm mới
    add_window = tk.Toplevel(top)
    add_window.title("Thêm biển báo")
    add_window.geometry('300x200')
    add_window.resizable(False, False)
    add_window.update_idletasks()
    x = (top.winfo_screenwidth() - add_window.winfo_reqwidth()) // 2
    y = (top.winfo_screenheight() - add_window.winfo_reqheight()) // 2
    add_window.geometry(f"+{x}+{y}")
    
    # Tạo frame chứa các controls
    form_frame = tk.Frame(add_window, padx=10, pady=10)
    form_frame.pack(fill=tk.BOTH, expand=True)
    
    tk.Label(form_frame, text="Chọn loại biển báo:", anchor="w").pack(fill=tk.X, pady=(0, 5))
    category_var = tk.StringVar()
    category_combobox = ttk.Combobox(form_frame, textvariable=category_var, 
                                   values=["Biển báo cấm", "Biển báo bắt buộc", "Biển báo nguy hiểm", "Biển báo khác"])
    category_combobox.pack(fill=tk.X, pady=(0, 10))
    category_combobox.current(0)
    
    tk.Label(form_frame, text="Nhập tên biển báo:", anchor="w").pack(fill=tk.X, pady=(0, 5))
    name_entry = tk.Entry(form_frame)
    name_entry.pack(fill=tk.X, pady=(0, 15))
    
    # Hiển thị thông tin dự đoán nếu có
    if max_prob > 0:
        prediction_text = f"Loại biển báo dự đoán: {max_prob*100:.1f}% tin cậy"
        tk.Label(form_frame, text=prediction_text, fg="blue").pack(fill=tk.X, pady=(0, 10))
    
    # Frame chứa các nút
    button_frame = tk.Frame(form_frame)
    button_frame.pack(fill=tk.X)
    
    def save_label():
        category = category_var.get()
        name = name_entry.get().strip()
        if not name:
            messagebox.showerror("Lỗi", "Tên biển báo không được để trống!")
            return 
        
        # Lấy lại dự đoán cho ảnh hiện tại
        processed_image = preprocess_image(selected_image_path)
        predicted_index, _ = predict_label(processed_image)
        
        # Tạo text label theo định dạng mong muốn
        label_text = f"{category}\n{name}"
        
        # Thêm label vào bảng labels
        label_id = insert_label(int(predicted_index), label_text)
        if label_id is None:
            return  # Thông báo lỗi đã được hiển thị trong insert_label()
        
        # Sau khi thêm label, thêm thông tin hình ảnh vào bảng images
        title = name  # Bạn có thể thay đổi theo ý muốn
        description = category  # Sử dụng loại biển báo làm mô tả
        insert_image(title, description, selected_image_path, label_id)
        
        messagebox.showinfo("Thông báo", "Thêm thành công!")
        add_window.destroy()
        show_classify_button(selected_image_path)
        
        # Cập nhật giao diện hiển thị biển báo
        label.configure(foreground='#011638', text=label_text)
        
        # Cập nhật danh sách labels và classNames
        global labels, classNames
        labels = get_labels_from_db()
        classNames = labels
        
        # Thông báo cho người dùng rằng có thể cần huấn luyện lại mô hình
        update_response = messagebox.askyesno("Lưu ý", "Biển báo đã được thêm vào cơ sở dữ liệu, nhưng mô hình cần được huấn luyện lại để nhận diện chính xác loại biển báo này trong tương lai.")

        if update_response:
            update_model()
        
    # Nút hủy
    cancel_btn = tk.Button(button_frame, text="Hủy", command=add_window.destroy, 
                         bg="#f44336", fg="white", padx=15)
    cancel_btn.pack(side=tk.RIGHT, padx=5)
    
    # Nút thêm
    add_btn = tk.Button(button_frame, text="Thêm", command=save_label, 
                       bg="#4CAF50", fg="white", padx=15)
    add_btn.pack(side=tk.RIGHT)
       
def open_edit_label():
    global current_image_path
    if not current_image_path:
        messagebox.showerror("Lỗi", "Cần tải ảnh lên trước khi sửa")
        return

    processed_image = preprocess_image(current_image_path)
    detected_index, _ = predict_label(processed_image)  # Sử dụng _ vì không cần max_prob

    labels = get_labels_from_db()
    if detected_index in labels:
        edit_label(detected_index, on_close=lambda: classify(current_image_path))
    else:
        messagebox.showerror("Lỗi", "Không tìm thấy biển báo để sửa!")
  
def reset_image():
    global current_image_path, classify_b, is_recognized

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

def edit_label(index, on_close=None):
    global labels
    
    if str(index) not in labels:
        messagebox.showerror("Lỗi", "Không tìm thấy biển báo để sửa!")
        return

    label_text = labels[str(index)]
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
        try:
            new_category = category_var.get()
            new_name = name_entry.get().strip()
            if not new_name:
                messagebox.showerror("Lỗi", "Tên biển báo không được để trống!")
                return

            # Cập nhật label trong database
            connection = connect_db()
            cursor = connection.cursor()
            new_label_text = f"{new_category}\n{new_name}"
            
            # Cập nhật bản ghi label trong database
            cursor.execute("UPDATE labels SET class_name = %s WHERE class_id = %s", 
                          (new_label_text, index))
            connection.commit()
            cursor.close()
            connection.close()
            
            # Cập nhật labels và classNames
            global labels, classNames
            labels = get_labels_from_db()  # Refresh labels từ database
            classNames = labels
            
            messagebox.showinfo("Thông báo", "Sửa thành công!")
            edit_window.destroy()
            if on_close:
                on_close()
        except mysql.connector.Error as e:
            messagebox.showerror("Lỗi", f"Không thể cập nhật: {str(e)}")
        
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
            messagebox.showinfo("Webcam", "Đang mở chế độ webcam. Nhấn 'Esc' để thoát khỏi chế độ webcam.")
        else:
            messagebox.showerror("Lỗi", f"Không tìm thấy file webcam.py tại {webcam_path}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể mở webcam: {str(e)}")
    finally:
        top.deiconify()  # Khôi phục cửa sổ hiện tại sau khi đóng webcam
   
frame_buttons = tk.Frame(top, bg=top["bg"])
frame_buttons.pack(side=BOTTOM,pady=50)

update_btn = Button(frame_buttons, text="Cập nhật", command=update_model, padx=15, pady=8, \
                   background='#28a745', foreground='white', font=('Arial', 10, 'bold'), \
                   relief="raised", bd=2, cursor="hand2")
update_btn.pack(side=LEFT, padx=12, pady=5) 

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