from alveoleye._workers import ProcessingWorker
import cv2


def test_config_load():
    worker = ProcessingWorker()
    assert 'TITLE' in worker.config_data['ProcessingActionBox'].keys()


def test_load_model():
    from alveoleye.lungcv.model_operations import init_trained_model
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    worker = ProcessingWorker()
    model = init_trained_model("src/alveoleye/data/default.pth")
    assert type(model) is MaskRCNN


def test_load_proxy_viewer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    assert type(viewer) is not None


def test_processing_flow(make_napari_viewer_proxy, qtbot):
    from alveoleye._widget import WidgetMain
    viewer = make_napari_viewer_proxy()
    widget = WidgetMain(viewer)
    widget.processing_group_box.import_paths["image"] = "src/alveoleye/_tests/test_images/1.tif"
    widget.processing_group_box.image = cv2.imread(widget.processing_group_box.import_paths["image"])
    box = widget.processing_group_box
    box.worker = ProcessingWorker()
    # Todo: use the actionbox click to do this
    worker = widget.processing_group_box.worker
    worker.set_napari_viewer(box.napari_viewer)
    worker.set_image_path(box.import_paths["image"])
    worker.set_weights(box.import_paths["weights"])
    worker.set_labels(box.labels_config_data)
    worker.set_image_shape(box.image.shape)
    worker.set_confidence_threshold_value(box.confidence_threshold_spin_box.value())
    with qtbot.waitSignal(worker.results_ready, timeout=50000):
        worker.results_ready.connect(box.on_results_ready)
        worker.run()

    assert len(widget.napari_viewer.layers) > 0
