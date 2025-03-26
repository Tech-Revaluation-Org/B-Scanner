import sys
import cv2
import numpy as np
import barcode
from barcode.writer import ImageWriter
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QMenuBar, QStatusBar,
    QLineEdit, QComboBox, QTabWidget, QFormLayout, QSpinBox, QSplitter, QMessageBox
)

class BarcodeScanner(QObject):
    frame_processed = pyqtSignal(QImage, list)
    error_occurred = pyqtSignal(str)

    def __init__(self, backend='opencv', parent=None):
        super().__init__(parent)
        self.backend = backend
        self._setup_detector()
        self.active = False
        self.cap = None
        self.frame_interval = 30  # milliseconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)

    def _setup_detector(self):
        try:
            if self.backend == 'opencv':
                self.detector = cv2.barcode_BarcodeDetector()
            elif self.backend == 'pyzbar':
                from pyzbar.pyzbar import decode
                self.detector = decode
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
        except Exception as e:
            self.error_occurred.emit(f"Detector init failed: {str(e)}")
            raise

    def start_capture(self, camera_index=0):
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise IOError("Camera not accessible")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.active = True
            self.timer.start(self.frame_interval)
            return True
        except Exception as e:
            self.error_occurred.emit(f"Camera error: {str(e)}")
            return False

    def process_next_frame(self):
        if not self.active or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            processed_frame, results = self.process_frame(frame)
            self.frame_processed.emit(self.convert_frame(processed_frame), results)

    def process_frame(self, frame):
        decoded = []
        try:
            frame = cv2.flip(frame, 1)
            
            if self.backend == 'opencv':
                retval, info, _, corners = self.detector.detectAndDecode(frame)
                if retval:
                    for data, pts in zip(info, corners):
                        if data:
                            decoded.append({
                                'data': data,
                                'polygon': pts.astype(np.int32).reshape((-1, 1, 2))
                            })
            elif self.backend == 'pyzbar':
                barcodes = self.detector(frame)
                for barcode in barcodes:
                    decoded.append({
                        'data': barcode.data.decode('utf-8'),
                        'polygon': np.array(barcode.polygon, np.int32)
                    })

            return frame, decoded
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")
            return frame, []

    def stop_capture(self):
        self.active = False
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def convert_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        return QImage(frame.data, w, h, bytes_per_line, 
                     QImage.Format.Format_BGR888)

class ScannerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.scanner = None
        self.current_backend = 'opencv'
        self.init_ui()
        self.setup_scanner()

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.video_display = QLabel("Camera Feed")
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setFixedSize(640, 480)
        self.video_display.setStyleSheet("border: 2px solid #555; background: #333")

        control_layout = QHBoxLayout()
        self.btn_toggle = QPushButton("Start Camera")
        self.btn_toggle.setFixedWidth(120)
        self.btn_toggle.clicked.connect(self.toggle_camera)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(['opencv', 'pyzbar'])
        self.backend_combo.currentTextChanged.connect(self.change_backend)
        self.status_label = QLabel("Status: Idle")
        self.status_label.setFont(QFont("Arial", 10))
        control_layout.addWidget(self.btn_toggle)
        control_layout.addWidget(self.backend_combo)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setFont(QFont("Courier", 10))
        
        button_layout = QHBoxLayout()
        self.btn_scan = QPushButton("Scan Image File")
        self.btn_scan.clicked.connect(self.scan_image)
        self.btn_scan.setFixedWidth(150)
        self.btn_clear = QPushButton("Clear Results")
        self.btn_clear.clicked.connect(self.results.clear)
        self.btn_clear.setFixedWidth(150)
        button_layout.addWidget(self.btn_scan)
        button_layout.addWidget(self.btn_clear)
        button_layout.addStretch()

        main_layout.addWidget(self.video_display)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.results)
        self.setLayout(main_layout)

    def setup_scanner(self):
        try:
            self.scanner = BarcodeScanner(self.current_backend)
            self.scanner_thread = QThread()
            self.scanner.moveToThread(self.scanner_thread)
            self.scanner.frame_processed.connect(self.update_display)
            self.scanner.error_occurred.connect(self.handle_error)
            self.scanner_thread.started.connect(self.scanner.start_capture)
            self.scanner_thread.start()
        except Exception as e:
            self.show_error(f"Scanner setup failed: {str(e)}")

    def toggle_camera(self):
        if self.scanner and self.scanner.active:
            self.scanner.stop_capture()
            self.btn_toggle.setText("Start Camera")
            self.status_label.setText("Status: Stopped")
        else:
            self.btn_toggle.setText("Stop Camera")
            self.status_label.setText("Status: Starting...")
            QTimer.singleShot(100, self.ensure_camera_started)

    def ensure_camera_started(self):
        if self.scanner.start_capture():
            self.status_label.setText("Status: Running")
        else:
            self.status_label.setText("Status: Failed")
            self.btn_toggle.setText("Start Camera")

    def change_backend(self, backend):
        if self.scanner and self.scanner.active:
            self.scanner.stop_capture()
        self.current_backend = backend
        self.setup_scanner()
        self.status_label.setText(f"Backend: {backend}")

    def update_display(self, image, results):
        pixmap = QPixmap.fromImage(image)
        self.video_display.setPixmap(pixmap.scaled(
            self.video_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        
        for result in results:
            if result['data'] not in self.results.toPlainText():
                self.results.append(f"Detected [{self.current_backend}]: {result['data']}")

    def scan_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                img = cv2.imread(path)
                if img is not None:
                    processed_img, results = self.scanner.process_frame(img)
                    self.update_display(self.scanner.convert_frame(processed_img), results)
                else:
                    self.show_error("Failed to load image")
            except Exception as e:
                self.show_error(f"Image scan failed: {str(e)}")

    def handle_error(self, message):
        self.show_error(message)
        self.scanner.stop_capture()
        self.btn_toggle.setText("Start Camera")
        self.status_label.setText("Status: Error")

    def show_error(self, message):
        self.results.append(f"<Error> {message}")

    def closeEvent(self, event):
        if self.scanner:
            self.scanner.stop_capture()
        if self.scanner_thread and self.scanner_thread.isRunning():
            self.scanner_thread.quit()
            self.scanner_thread.wait()
        super().closeEvent(event)

class BarcodeGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter barcode text")
        self.format_combo = QComboBox()
        self.format_combo.addItems(['EAN13', 'Code128', 'QR'])
        
        form_layout.addRow("Text:", self.text_input)
        form_layout.addRow("Format:", self.format_combo)
        
        self.generate_btn = QPushButton("Generate Barcode")
        self.generate_btn.clicked.connect(self.generate_barcode)
        
        self.barcode_display = QLabel("Barcode will appear here")
        self.barcode_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.barcode_display.setFixedSize(400, 200)
        self.barcode_display.setStyleSheet("border: 1px solid #555")
        
        self.save_btn = QPushButton("Save Barcode")
        self.save_btn.clicked.connect(self.save_barcode)
        self.save_btn.setEnabled(False)
        
        layout.addLayout(form_layout)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.barcode_display)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)
        
        self.generated_image = None

    def generate_barcode(self):
        text = self.text_input.text().strip()
        if not text:
            return
        
        format_map = {
            'EAN13': 'ean13',
            'Code128': 'code128',
            'QR': 'qrcode'
        }
        b_type = format_map[self.format_combo.currentText()]
        
        try:
            if b_type == 'qrcode':
                barcode_instance = barcode.get('qrcode', text, writer=ImageWriter())
            else:
                barcode_instance = barcode.get(b_type, text, writer=ImageWriter())
            
            filename = f"temp_barcode.{barcode_instance.writer.format.lower()}"
            with open(filename, 'wb') as f:
                barcode_instance.write(f)
            
            pixmap = QPixmap(filename)
            self.barcode_display.setPixmap(pixmap.scaled(
                self.barcode_display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            self.generated_image = filename
            self.save_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Barcode generation failed:\n{str(e)}")

    def save_barcode(self):
        if not self.generated_image:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Barcode", "", f"Images (*.{self.format_combo.currentText().lower()})"
        )
        if path:
            try:
                from shutil import copyfile
                copyfile(self.generated_image, path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed:\n{str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Barcode Toolkit")
        self.setGeometry(100, 100, 1200, 800)
        
        self.tabs = QTabWidget()
        self.scanner_widget = ScannerWidget()
        self.generator_widget = BarcodeGenerator()
        
        self.tabs.addTab(self.scanner_widget, "Scanner")
        self.tabs.addTab(self.generator_widget, "Generator")
        
        self.setCentralWidget(self.tabs)
        
        self.init_menu()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def init_menu(self):
        menu_bar = QMenuBar()
        
        file_menu = menu_bar.addMenu("&File")
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        self.setMenuBar(menu_bar)

    def show_about(self):
        about_text = """
        <h2>Barcode Toolkit v1.2</h2>
        <p>Developed with PyQt6 and OpenCV</p>
        <p>Features:</p>
        <ul>
            <li>Real-time barcode scanning</li>
            <li>Image file scanning</li>
            <li>Barcode generation (EAN13, Code128, QR)</li>
            <li>Multiple detection backends</li>
        </ul>
        """
        QMessageBox.about(self, "About", about_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    cv2.setNumThreads(4)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
