from utils import get_toy_data_image_classification, get_automl_settings

def test_image_classification():
    from flaml import AutoML

    img_train, label_train, img_val, label_val, imf_test = get_toy_data_image_classification()
    autoML = AutoML()
    automl_settings = get_automl_settings()

    automl_settings["task"] = "image_classification"
    automl_settings["metric"] = "accuracy"

if __name__ == "__main__":
    test_image_classification()