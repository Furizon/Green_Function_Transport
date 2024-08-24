import sys
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QCloseEvent
from GFtransport import *
from queue import Queue
calculation_finished = False
q = Queue(maxsize=0)

class UIMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        
        self.file_name = ""
        self.file_type = ""

        self.setWindowTitle("HubbardUI")
        self.menu_bar = QtWidgets.QMenuBar()
        self.main_layout = QtWidgets.QVBoxLayout()
        self.top_layout = QtWidgets.QHBoxLayout()
        self.bottom_layout = QtWidgets.QVBoxLayout()
        
        self.load_button = QtWidgets.QPushButton("Select Config File")
        self.run_button = QtWidgets.QPushButton("Run")
        self.output_text = QtWidgets.QTextEdit(self, readOnly = True)
        self.output_text.ensureCursorVisible()
        self.output_text.setLineWrapColumnOrWidth(960)
        self.output_text.setLineWrapMode(QtWidgets.QTextEdit.FixedPixelWidth)
        self.output_text.setStyleSheet('border: 1px solid black')
        
        self.image_label = QtWidgets.QLabel(self)

        self.input_path = QtWidgets.QLabel("")
        self.input_path.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.input_path.setStyleSheet('border: 1px solid black')

        self.save_button = QtWidgets.QPushButton("Save Image")

        self.top_layout.addWidget(self.output_text)
        self.top_layout.addWidget(self.image_label)
        
        self.bottom_layout.addWidget(self.load_button)
        self.bottom_layout.addWidget(self.input_path)
        self.bottom_layout.addWidget(self.run_button)
        self.bottom_layout.addWidget(self.save_button)

        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.bottom_layout)

        self.main_widget = QtWidgets.QWidget()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.load_button.clicked.connect(self.readFile)
        self.run_button.clicked.connect(self.runCode)
        self.save_button.clicked.connect(self.SaveImage)

        self.worker = Worker()
        self.worker.text_signal.connect(self.onUpdateText)
        sys.stdout = self.worker


    def onUpdateText(self, text):
        cursor = self.output_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.output_text.setTextCursor(cursor)
        self.output_text.ensureCursorVisible()
        if text == "***Showing the Tranmission Spectrum***":
            canvas = q.get(block = True)
            pixmap = QtGui.QPixmap(canvas.grab().toImage())
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            


    def runCode(self):
        if self.input_path.text() != "" and self.input_path.text():
            print("Run " + self.input_path.text())
            q.put(self.input_path.text(), block = True)
            worker = Worker(parent=self)
            worker.start()

    def readFile(self):
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(self, "Select Config File", "./", "*.*")
        if file_name:
            print("Load File " + file_name)
            self.input_path.setText(file_name)
            LoadHamiltonianTxt(file_name)

    def SaveImage(self):
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_label.pixmap().save(file_name)
            print(file_name + " Saved")
        

    def closeEvent(self, event: QCloseEvent) -> None:
        sys.stdout = sys.__stdout__
        return super().closeEvent(event)



class Worker(QtCore.QThread):

    finished_signal = QtCore.Signal(str)
    text_signal = QtCore.Signal(str)
    
    def __init__(self, data = None, parent = None):
        super(Worker, self).__init__(parent)
        self.data = data

    def run(self):
        path = q.get(block = True)
        print(path)
        canvas, _ = hubbardMethodCalTransmission(path)
        q.put(canvas, block = True)
        self.finished.emit()

    def write(self, text):
        self.text_signal.emit(str(text))
    
    def flush(self):
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    main_UI = UIMainWindow()
    main_UI.resize(1280, 720)
    main_UI.show()

    sys.exit(app.exec())