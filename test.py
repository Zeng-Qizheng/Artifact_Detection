import sys
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtUiTools import QUiLoader
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

app = QtWidgets.QApplication(sys.argv)
loader = QUiLoader()
window = loader.load("pyqtgraph_window.ui")
window.graphWidget.plot([1,5,10], [1,2,7])

window.show()
app.exec_()



if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    loader = UiLoader()
    window = loader.load(os.path.join(CURRENT_DIR, "pyqtgraph_window.ui"))
    window.graphWidget.plot([1, 5, 10], [1, 2, 7])
    window.show()
    sys.exit(app.exec_())