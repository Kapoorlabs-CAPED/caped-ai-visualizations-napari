import numpy as np

from caped_ai_visualizations_napari import (
    plugin_wrapper_caped_ai_visualization,
)


# make_napari_viewer is a pytest fixture that returns a napari viewer object
def test_example_q_widget(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    plugin_wrapper_caped_ai_visualization(viewer)
