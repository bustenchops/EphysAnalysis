import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMenu


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

    def contextMenuEvent(self, e):
        context = QMenu(self)
        crap = QAction("thisone", self)
        crap.triggered.connect(self.fuckthisshit)
        context.addAction(crap)
        context.addAction(QAction("test 2", self))
        context.addAction(QAction("test 3", self))
        context.exec_(e.globalPos())

    def fuckthisshit(selfself):
        print('jesus ever fucking christ')



app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()