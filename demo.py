import cv2
import gradio as gr

from main import clahe_preprocess, get_model


def run_inference(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = get_model()(clahe_preprocess(bgr))[0]
    return cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)


gr.Interface(
    run_inference, "image", "image", title="Dental Pathology Detector"
).launch()
