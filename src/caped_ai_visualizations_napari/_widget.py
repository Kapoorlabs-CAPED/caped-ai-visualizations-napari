"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from napari import Viewer
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CapeVisuFrameWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self):
        super().__init__()

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._layout.addWidget(
            QLabel("caped-ai visualization wizard", parent=self)
        )

        self.figure = plt.figure(figsize=(4, 4))
        self.multiplot_widget = FigureCanvas(self.figure)
        self.multiplot_widget.setMinimumSize(200, 200)
        self.ax = self.multiplot_widget.figure.subplots(1, 1)

        self._layout.addWidget(self.multiplot_widget)

        # Add network parameter json dropdown menu

        self.paramjsonbox = QPushButton("Open network parameter json file")

        self._layout.addWidget(self.paramjsonbox)

        # Add coordinate json dropdown menu

        self.cordjsonbox = QPushButton("Open coordinates json file")

        self._layout.addWidget(self.cordjsonbox)
        # Add catagories json dropdown menu

        self.catjsonbox = QPushButton("Open categories json file")

        self._layout.addWidget(self.catjsonbox)


class CapeVisuWidget(QWidget):
    def __init__(
        self,
        viewer: Viewer,
    ):

        self.viewer = viewer
        # Connnectors
        self.frameWidget = CapeVisuFrameWidget()
        self.frameWidget.paramjsonbox.clicked.connect(
            self._capture_paramjson_callback()
        )
        self.frameWidget.cordjsonbox.clicked.connect(
            self._capture_cordjson_callback()
        )
        self.frameWidget.catjsonbox.clicked.connect(
            self._capture_catjson_callback()
        )

    def _capture_paramjson_callback(self):

        dialog = QFileDialog()
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Open network Parameter json file (*.json)")
        self.paramfile = dialog.selectFile(filename=dialog.getOpenFileName())

    def _capture_cordjson_callback(self):

        dialog = QFileDialog()
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Open coordinates json file (*.json)")

        self.cordfile = dialog.selectFile(filename=dialog.getOpenFileName())

    def _capture_catjson_callback(self):

        dialog = QFileDialog()
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Open categories json file (*.json)")
        self.catfile = dialog.selectFile(filename=dialog.getOpenFileName())
