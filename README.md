
# 🎨 Flask Image Filter App

A Flask-based web application that allows users to apply over 100 different image filters to their selected images — either through file upload or live camera feed. The project is divided into two main sections:

## 🔹 Features

### 1. Static Image Filters
- Upload an image from your local device.
- Choose from 100+ filters (e.g., grayscale, blur, sharpen, artistic effects).
- Instantly preview how the selected filter will look on your image.
- Download the filtered image.

### 2. Live Camera Filter (Real-Time)
- Activate your device camera (e.g., laptop webcam).
- Choose a live filter to apply in real-time.
- Watch the effect on your camera feed instantly — great for testing filters on live video!

## ⚙️ Technologies & Libraries Used

This project uses the following technologies and Python libraries:

- **Flask** – Web framework for backend
- **OpenCV (cv2)** – Image processing & live camera support
- **Pillow (PIL)** – Image handling and processing
- **ImageFilter** – Built-in PIL module for applying standard filters
- **HTML5 / CSS / JavaScript** – Frontend and live preview
- **Webcam API** – For accessing device camera in the browser

## 🧱 Architecture

This project follows a **separated backend and frontend architecture**:

- **Backend** (Flask) handles the image processing and filter application logic.
- **Frontend** (served with `http.server`) manages the user interface, camera access, and displays the output.

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Required libraries:
  ```bash
  pip install flask pillow opencv-python
  ```

### Run the App

You need to start both the backend and frontend separately using two terminals:

#### 🖥️ Start the Backend (Flask API):

```bash
python app.py
```

This will start the Flask server on:

```
http://localhost:5000
```

#### 🌐 Start the Frontend (Static Files Server):

Open a **new terminal window or tab** and run:

```bash
python -m http.server 8000
```

This will serve the frontend at:

```
http://localhost:8000
```

Now, you can open the frontend URL in your browser and interact with the app!

## 📁 Project Structure

```
.
├── app.py                  # Flask backend logic
├── static/
│   ├── filters/            # Filter scripts / files
│   ├── js/                 # Frontend JS (camera logic, filters)
│   └── ...
├── templates/
│   └── index.html          # Main interface
└── README.md               # You're reading it!
```

## 📷 Demo (Optional)
You can add screenshots or GIFs of the app in action here for better visual reference.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

---

Feel free to contribute or suggest improvements!
