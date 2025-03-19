import sys
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap, QColor
from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, pyqtSlot

# --- Qt3D imports for 3D scene ---
from PyQt6.Qt3DExtras import (
    Qt3DWindow, QOrbitCameraController, QCuboidMesh, QPhongMaterial
)
from PyQt6.Qt3DCore import QEntity, QTransform

# ---------- Video Scanner Widget ----------
class VideoScannerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(0)  # Open default camera
        if not self.cap.isOpened():
            raise Exception("Could not open camera.")
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # about 30 FPS

    def initUI(self):
        layout = QVBoxLayout()
        self.video_label = QLabel("Camera feed not available", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        self.results_box = QTextEdit(self)
        self.results_box.setReadOnly(True)
        layout.addWidget(self.results_box)

        btn_layout = QHBoxLayout()
        self.load_img_btn = QPushButton("Scan from Image", self)
        self.load_img_btn.clicked.connect(self.scan_from_image)
        btn_layout.addWidget(self.load_img_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    @pyqtSlot()
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame (flip horizontally for a mirror effect)
            frame = cv2.flip(frame, 1)
            barcodes = decode(frame)
            # Draw bounding boxes and display decoded text
            for barcode in barcodes:
                (x, y, w, h) = barcode.rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                barcode_data = barcode.data.decode("utf-8")
                barcode_type = barcode.type
                text = f"{barcode_data} ({barcode_type})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
                # Append result to the text box if not already shown
                if self.results_box.toPlainText().find(text) == -1:
                    self.results_box.append(text)

            # Convert frame to QImage and display
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def scan_from_image(self):
        # Let user choose an image file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                self.results_box.append("Error: Cannot load image.")
                return
            barcodes = decode(img)
            results = []
            for barcode in barcodes:
                barcode_data = barcode.data.decode("utf-8")
                barcode_type = barcode.type
                results.append(f"{barcode_data} ({barcode_type})")
            if results:
                self.results_box.append("Image scan results:")
                self.results_box.append("\n".join(results))
            else:
                self.results_box.append("No barcode found in the image.")

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

# ---------- 3D Widget using Qt3D ----------
class ThreeDWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create a Qt3DWindow and embed it into this widget
        self.view = Qt3DWindow()
        self.container = self.createWindowContainer(self.view, self)
        self.container.setMinimumSize(400, 400)
        self.container.setFocusPolicy(Qt.FocusPolicy.TabFocus)

        # Create the 3D scene
        self.scene = QEntity()
        self.create_cube(self.scene)
        self.view.setRootEntity(self.scene)

        # Set up camera controls
        self.camera = self.view.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16/9, 0.1, 1000)
        self.camera.setPosition(QVector3D(0, 0, 20))
        self.camera.setViewCenter(QVector3D(0, 0, 0))
        cam_controller = QOrbitCameraController(self.scene)
        cam_controller.setCamera(self.camera)

        layout = QVBoxLayout()
        layout.addWidget(self.container)
        self.setLayout(layout)

    def create_cube(self, root_entity):
        # Create a simple cuboid mesh
        cube_entity = QEntity(root_entity)
        mesh = QCuboidMesh()
        cube_entity.addComponent(mesh)

        # Create a transform that rotates the cube continuously
        self.transform = QTransform()
        cube_entity.addComponent(self.transform)

        # Apply a material
        material = QPhongMaterial(root_entity)
        material.setDiffuse(QColor(100, 100, 255))
        cube_entity.addComponent(material)

        # Animate the cube (rotation animation)
        self.anim = QPropertyAnimation(self.transform, b"rotationAngle")
        self.anim.setDuration(5000)
        self.anim.setStartValue(0)
        self.anim.setEndValue(360)
        self.anim.setLoopCount(-1)
        self.anim.start()

# ---------- Main Window Combining Both Widgets ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Barcode Scanner")
        self.resize(1200, 700)

        # Main container widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        # Left side: Video scanner with barcode detection
        self.scanner_widget = VideoScannerWidget(self)
        layout.addWidget(self.scanner_widget, 2)

        # Right side: 3D scene widget
        self.three_d_widget = ThreeDWidget(self)
        layout.addWidget(self.three_d_widget, 1)

        # Optional: add a refresh button to clear results
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Results", self)
        self.clear_btn.clicked.connect(self.clear_results)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

    def clear_results(self):
        self.scanner_widget.results_box.clear()

    def closeEvent(self, event):
        # Ensure the video capture is released
        self.scanner_widget.closeEvent(event)
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
