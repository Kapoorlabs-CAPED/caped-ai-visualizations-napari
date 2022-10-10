"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari


class CapeVisuWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        csvdir: str,
        parameter_json: str,
        coordinate_json: str,
        catagories_json: str,
        start_project_mid=0,
        end_project_mid=1,
        event_count_plot="Plot selected event count",
        cell_count_plot="Plot cell count",
        event_norm_count_plot="Plot selected normalized event count",
    ):
        super().__init__()

        self._layout = QVBoxLayout()
        self._outer_scroll = QScrollArea()
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

        self.paramjsonbox = QFileDialog()
        self.paramjsonbox.setAcceptMode(QFileDialog.AcceptOpen)
        self.paramjsonbox.setFileMode(QFileDialog.ExistingFile)
        self.paramjsonbox.setFilter(
            "Open network Parameter json file (*.json)"
        )

        self._layout.addWidget(self.paramjsonbox)

        # Add coordinate json dropdown menu

        self.cordjsonbox = QFileDialog()
        self.cordjsonbox.setAcceptMode(QFileDialog.AcceptOpen)
        self.cordjsonbox.setFileMode(QFileDialog.ExistingFile)
        self.cordjsonbox.setFilter("Open coordinates json file (*.json)")

        self._layout.addWidget(self.cordjsonbox)
        # Add catagories json dropdown menu

        self.catjsonbox = QFileDialog()
        self.catjsonbox.setAcceptMode(QFileDialog.AcceptOpen)
        self.catjsonbox.setFileMode(QFileDialog.ExistingFile)
        self.catjsonbox.setFilter("Open categories json file (*.json)")

        self._layout.addWidget(self.catjsonbox)

        self._outer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._outer_scroll.setWidgetResizable(True)
        self._outer_scroll.setWidget(self)

        # Connnectors

        self.paramjsonbox.filesSelected.connect(
            self._capture_paramjson_callback()
        )
        self.cordjsonbox.filesSelected.connect(
            self._capture_cordjson_callback()
        )
        self.catjsonbox.filesSelected.connect(self._capture_catjson_callback())

    def _capture_paramjson_callback(self):

        self.paramfile = self.paramjsonbox.selectFile()

    def _capture_cordjson_callback(self):

        self.cordfile = self.cordjsonbox.selectFile()

    def _capture_catjson_callback(self):

        self.catfile = self.catjsonbox.selectFile()
