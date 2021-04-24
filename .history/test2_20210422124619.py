import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
sys.coinit_flags = 2
import pywinauto
from PyQt5.QtWidgets import QApplication, QMainWindow


def main():
    app = QApplication(sys.argv)
    w = QMainWindow()
    w.show()
    app.exec_()


if __name__ == '__main__':
    main()