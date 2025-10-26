import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QComboBox, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from deepface import DeepFace

class DeepFaceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DeepFace GUI')
        self.setGeometry(100, 100, 800, 600)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Image 1 selection
        self.img1_label = QLabel('Image 1: None')
        layout.addWidget(self.img1_label)
        load_img1_btn = QPushButton('Load Image 1')
        load_img1_btn.clicked.connect(self.load_image1)
        layout.addWidget(load_img1_btn)

        # Image 2 selection (for verification)
        self.img2_label = QLabel('Image 2: None')
        layout.addWidget(self.img2_label)
        load_img2_btn = QPushButton('Load Image 2')
        load_img2_btn.clicked.connect(self.load_image2)
        layout.addWidget(load_img2_btn)

        # Function selection
        self.function_combo = QComboBox()
        self.function_combo.addItems(['Verify Faces', 'Analyze Face'])
        layout.addWidget(QLabel('Select Function:'))
        layout.addWidget(self.function_combo)

        # Execute button
        execute_btn = QPushButton('Execute')
        execute_btn.clicked.connect(self.execute_function)
        layout.addWidget(execute_btn)

        # Result display
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(QLabel('Results:'))
        layout.addWidget(self.result_text)

        # Image display area
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_display)

        # Variables for images
        self.img1_path = None
        self.img2_path = None

    def load_image1(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image 1", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.img1_path = file_path
            self.img1_label.setText(f'Image 1: {file_path}')
            self.display_image(file_path)

    def load_image2(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image 2", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.img2_path = file_path
            self.img2_label.setText(f'Image 2: {file_path}')
            self.display_image(file_path)

    def display_image(self, path):
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
        self.image_display.setPixmap(scaled_pixmap)

    def execute_function(self):
        function = self.function_combo.currentText()
        if not self.img1_path:
            QMessageBox.warning(self, 'Warning', 'Please load Image 1.')
            return

        try:
            if function == 'Verify Faces':
                if not self.img2_path:
                    QMessageBox.warning(self, 'Warning', 'Please load Image 2 for verification.')
                    return
                result = DeepFace.verify(self.img1_path, self.img2_path)
                self.result_text.setPlainText(f"Verification Result: {result}")
            elif function == 'Analyze Face':
                result = DeepFace.analyze(self.img1_path)
                self.result_text.setPlainText(f"Analysis Result: {result}")
        except Exception as e:
            self.result_text.setPlainText(f"Error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DeepFaceGUI()
    window.show()
    sys.exit(app.exec_())
