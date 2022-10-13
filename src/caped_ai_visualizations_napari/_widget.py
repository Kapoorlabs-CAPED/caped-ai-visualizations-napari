"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

import functools
import time
from pathlib import Path
from typing import List

import napari
import numpy as np
from magicgui import magicgui
from magicgui import widgets as mw
from napari.qt.threading import thread_worker
from psygnal import Signal
from qtpy.QtWidgets import QSizePolicy, QTabWidget, QVBoxLayout, QWidget


def plugin_wrapper_caped_ai_visualization():

    from oneat.NEATModels.neat_dynamic_resnet import NEATTResNet
    from oneat.NEATModels.neat_lstm import NEATLRNet
    from oneat.NEATModels.neat_static_resnet import NEATResNet
    from oneat.NEATModels.neat_vollnet import NEATVollNet
    from oneat.NEATUtils.utils import load_json
    from oneat.pretrained import get_model_folder, get_registered_models

    DEBUG = True

    def get_data(image, debug=DEBUG):

        image = image.data[0] if image.multiscale else image.data
        if debug:
            print("image loaded")
        return np.asarray(image)

    def abspath(root, relpath):
        root = Path(root)
        if root.is_dir():
            path = root / relpath
        else:
            path = root.parent / relpath
        return str(path.absolute())

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:
                    print(f"{str(emitter.name).upper()}: {source.name}")
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    _models_vollnet, _aliases_vollnet = get_registered_models(NEATVollNet)

    _models_lrnet, _aliases_lrnet = get_registered_models(NEATLRNet)

    _models_tresnet, _aliases_tresnet = get_registered_models(NEATTResNet)

    _models_resnet, _aliases_resnet = get_registered_models(NEATResNet)

    models_vollnet = [
        ((_aliases_vollnet[m][0] if len(_aliases_vollnet[m]) > 0 else m), m)
        for m in _models_vollnet
    ]
    print(models_vollnet)
    models_lrnet = [
        ((_aliases_lrnet[m][0] if len(_aliases_lrnet[m]) > 0 else m), m)
        for m in _models_lrnet
    ]

    models_tresnet = [
        ((_aliases_tresnet[m][0] if len(_aliases_tresnet[m]) > 0 else m), m)
        for m in _models_tresnet
    ]

    models_resnet = [
        ((_aliases_resnet[m][0] if len(_aliases_resnet[m]) > 0 else m), m)
        for m in _models_resnet
    ]

    nms_algorithms = ["iou"]

    model_parameters = dict()
    model_catagories = dict()
    model_cord = dict()

    model_selected = None
    worker = None
    CUSTOM_NEAT = "CUSTOM_NEAT"
    CSV_PREDICTIONS = "CSV_PREDICTIONS"
    PRETRAINED = "PRETRAINED"

    DEFAULTS_MODEL = dict(
        oneat_model_class=NEATVollNet,
        oneat_model_type=CUSTOM_NEAT,
        model_vollnet=models_vollnet[0][0],
        model_lrnet=models_lrnet[0][0],
        model_tresnet=models_tresnet[0][0],
        model_resnet=models_resnet[0][0],
        axes="TZYX",
    )

    oneat_model_class_choices = [
        ("TZYX", NEATVollNet),
        ("TYXC", NEATLRNet),
        ("YXT", NEATTResNet),
        ("YXC", NEATResNet),
    ]

    oneat_model_type_choices = [
        ("PreTrained", PRETRAINED),
        ("Load CSV", CSV_PREDICTIONS),
        ("Custom Oneat", CUSTOM_NEAT),
    ]

    DEFAULTS_PRED_PARAMETERS = dict(
        norm_image=True,
        n_tiles=(1, 1, 1),
        event_threshold=0.9,
        event_confidence=0.9,
        nms_function=nms_algorithms[0],
        start_project_mid=4,
        end_project_mid=1,
    )

    @magicgui(
        norm_image=dict(
            widget_type="CheckBox",
            text="Normalize Image",
            value=DEFAULTS_PRED_PARAMETERS["norm_image"],
        ),
        n_tiles=dict(
            widget_type="LiteralEvalLineEdit",
            label="Number of Tiles",
            value=DEFAULTS_PRED_PARAMETERS["n_tiles"],
        ),
        nms_function=dict(
            widget_type="ComboBox",
            label="Choice of non maximal supression algorithm",
            choices=nms_algorithms,
            value=DEFAULTS_PRED_PARAMETERS["nms_function"],
        ),
        event_threshold=dict(
            widget_type="FloatSpinBox",
            label="Score Threshold",
            min=0.0,
            max=1.0,
            step=0.05,
            value=DEFAULTS_PRED_PARAMETERS["event_threshold"],
        ),
        event_confidence=dict(
            widget_type="FloatSpinBox",
            label="Event Confidence",
            min=0.0,
            max=1.0,
            step=0.05,
            value=DEFAULTS_PRED_PARAMETERS["event_confidence"],
        ),
        start_project_mid=dict(
            widget_type="SpinBox",
            label="Start Project Mid",
            value=DEFAULTS_PRED_PARAMETERS["start_project_mid"],
        ),
        end_project_mid=dict(
            widget_type="SpinBox",
            label="End Project Mid",
            value=DEFAULTS_PRED_PARAMETERS["end_project_mid"],
        ),
        defaults_prediction_parameters_button=dict(
            widget_type="PushButton",
            text="Restore Prediction Parameters Defaults",
        ),
        call_button=False,
    )
    def plugin_prediction_parameters(
        norm_image,
        n_tiles,
        nms_function,
        event_threshold,
        event_confidence,
        start_project_mid,
        end_project_mid,
        defaults_prediction_parameters_button,
    ):

        return plugin_prediction_parameters

    kapoorlogo = abspath(__file__, "resources/kapoorlogo.png")
    citation = Path("https://doi.org/10.1242/focalplane.10759")

    @magicgui(
        label_head=dict(
            widget_type="Label",
            label=f'<h1> <img src="{kapoorlogo}"> </h1>',
            value=f'<h5><a href=" {citation}"> FocalPlane</a></h5>',
        ),
        image=dict(label="Input Image"),
        oneat_model_class=dict(
            widget_type="RadioButtons",
            label="Oneat Model Class",
            orientation="horizontal",
            choices=oneat_model_class_choices,
            value=DEFAULTS_MODEL["oneat_model_class"],
        ),
        oneat_model_type=dict(
            widget_type="RadioButtons",
            label="Oneat Model Type",
            orientation="horizontal",
            choices=oneat_model_type_choices,
            value=DEFAULTS_MODEL["oneat_model_type"],
        ),
        model_vollnet=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained VollNet Model",
            choices=models_vollnet,
            value=DEFAULTS_MODEL["model_vollnet"],
        ),
        model_lrnet=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained LRNet Model",
            choices=models_lrnet,
            value=DEFAULTS_MODEL["model_lrnet"],
        ),
        model_tresnet=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained TresNet Model",
            choices=models_tresnet,
            value=DEFAULTS_MODEL["model_tresnet"],
        ),
        model_resnet=dict(
            widget_type="ComboBox",
            visible=False,
            label="Pre-trained ResNet Model",
            choices=models_resnet,
            value=DEFAULTS_MODEL["model_resnet"],
        ),
        model_folder=dict(
            widget_type="FileEdit",
            visible=False,
            label="Custom Oneat",
            mode="r",
        ),
        csv_folder=dict(
            widget_type="FileEdit",
            visible=False,
            label="Load Oneat Detections",
            mode="r",
        ),
        defaults_model_button=dict(
            widget_type="PushButton", text="Restore Model Defaults"
        ),
        progress_bar=dict(label=" ", min=0, max=0, visible=False),
        layout="vertical",
        persist=True,
        call_button=True,
    )
    def plugin(
        viewer: napari.Viewer,
        label_head,
        image: napari.layers.Image,
        oneat_model_class,
        oneat_model_type,
        model_vollnet,
        model_lrnet,
        model_tresnet,
        model_resnet,
        model_folder,
        csv_folder,
        defaults_model_button,
        progress_bar: mw.ProgressBar,
    ) -> List[napari.types.LayerDataTuple]:
        # x = get_data(image)

        nonlocal worker

    """
    widget_for_modeltype = {
        NEATVollNet: plugin.model_vollnet,
        NEATLRNet: plugin.model_lrnet,
        NEATTResNet: plugin.model_tresnet,
        NEATResNet: plugin.model_resnet,
        CSV_PREDICTIONS: plugin.csv_folder,
        CUSTOM_NEAT: plugin.model_folder,
    }
    """
    tabs = QTabWidget()

    parameter_tab = QWidget()
    _parameter_tab_layout = QVBoxLayout()
    parameter_tab.setLayout(_parameter_tab_layout)
    _parameter_tab_layout.addWidget(plugin_prediction_parameters.native)
    tabs.addTab(parameter_tab, "Detection Parameter Selection")

    plugin.native.layout().addWidget(tabs)
    plugin.label_head.native.setOpenExternalLinks(True)
    plugin.label_head.native.setSizePolicy(
        QSizePolicy.MinimumExpanding, QSizePolicy.Fixed
    )

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active

    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet(
                "" if valid else "background-color: red"
            )

    class Updater:
        def __init__(self):
            def __call__(self, model_param, catconfig, cordconfig):
                setattr(self.model_param, model_param)
                setattr(self.catconfig, catconfig)
                setattr(self.cordconfig, cordconfig)
                self.viewer = plugin.viewer.value

            def _model(valid):
                widgets_valid(
                    plugin.oneat_model_class,
                    plugin.oneat_model_type,
                    plugin.model_folder.line_edit,
                    plugin.csv_folder.line_eit,
                    valid=valid,
                )

                if valid:
                    parameters = self.model_param
                    catagories = self.catconfig
                    cord = self.cordconfig

                    plugin.model_folder.line_edit.tooltip = ""
                    plugin.csv_folder.line_edit.tooltip = ""

                    return parameters, catagories, cord
                else:
                    plugin.model_folder.line_edit.tooltip = (
                        "Invalid model directory"
                    )

            def _n_tiles(valid):
                n_tiles, image, err = getattr(self.args, "n_tiles", (1, 1, 1))
                widgets_valid(
                    plugin_prediction_parameters.n_tiles,
                    valid=(valid or image is None),
                )
                if valid:
                    plugin_prediction_parameters.n_tiles.tooltip = "\n".join(
                        [
                            f"{t}: {s}"
                            for t, s in zip(n_tiles, get_data(image).shape)
                        ]
                    )
                    return n_tiles
                else:
                    msg = str(err) if err is not None else ""
                    plugin_prediction_parameters.n_tiles.tooltip = msg

            all_valid = True
            plugin.call_button.enabled = all_valid

    update = Updater()

    def select_model(key):
        nonlocal model_selected
        if key is not None:
            model_selected = key
            model_param = model_parameters.get(key)
            catconfig = model_catagories.get(key)
            cordconfig = model_cord.get(key)
            update(
                model_param,
                catconfig,
                cordconfig,
            )

    @change_handler(plugin.model_folder, init=False)
    def _model_folder_change(_path: str):
        path = Path(_path)
        key = CUSTOM_NEAT, path
        try:
            if not path.is_dir():
                return
            model_parameters[key] = load_json(str(path / "parameters.json"))
            model_catagories[key] = load_json(str(path / "catagories.json"))
            model_cord[key] = load_json(str(path / "cord.json"))
        except FileNotFoundError:
            pass
        finally:
            select_model(key)

    @change_handler(
        plugin.model_vollnet,
        plugin.model_lrnet,
        plugin.model_tresnet,
        plugin.model_resnet,
        init=False,
    )
    def _model_change(model_name: str):

        model_class = (
            NEATVollNet
            if Signal.sender() is plugin.model_vollnet
            else NEATLRNet
            if Signal.sender() is plugin.model_lrnet
            else NEATTResNet
            if Signal.sender() is plugin.model_tresnet
            else NEATResNet
            if Signal.sender() is plugin.model_resnet
            else None
        )
        key = model_class, model_name
        if key not in model_parameters:

            @thread_worker
            def _get_model_folder():
                return get_model_folder(*key)

            def _process_model_folder(path):
                try:
                    model_parameters[key] = load_json(
                        str(path / "parameters.json")
                    )
                    model_catagories[key] = load_json(
                        str(path / "catagories.json")
                    )
                    model_cord[key] = load_json(str(path / "cord.json"))
                finally:
                    select_model(key)
                    plugin.progress_bar.hide()

            worker = _get_model_folder()
            worker.returned.connect(_process_model_folder)
            worker.start()

            time.sleep(0.1)
            plugin.call_button.enabled = False
            plugin.progress_bar.label = "Downloading model"
            plugin.progress_bar.show()

        else:
            select_model(key)

    @change_handler(plugin_prediction_parameters.event_threshold)
    def _event_threshold_change(value: float):
        plugin_prediction_parameters.event_threshold.value = value

    @change_handler(plugin_prediction_parameters.event_confidence)
    def _event_confidence_change(value: float):
        plugin_prediction_parameters.event_confidence.value = value

    @change_handler(plugin.defaults_model_button, init=False)
    def restore_model_defaults():
        for k, v in DEFAULTS_MODEL.items():
            getattr(plugin, k).value = v

    @change_handler(plugin.image, init=False)
    def _image_change(image: napari.layers.Image):
        plugin.image.tooltip = (
            f"Shape: {get_data(image).shape, str(image.name)}"
        )

    @change_handler(
        plugin_prediction_parameters.defaults_prediction_parameters_button,
        init=False,
    )
    def restore_prediction_parameters_defaults():
        for k, v in DEFAULTS_MODEL.items():
            getattr(plugin, k).value = v

    return plugin
