import time
import cv2
import numpy as np
import onnxruntime  # type: ignore
from PIL import Image
from image_proc import refine_foreground
from typing import Optional
from functools import wraps

def time_it(func):
    """
    A decorator to measure the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Use perf_counter for high-resolution timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to execute.")
        return result
    return wrapper

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

@time_it
def run_inference(
    session: onnxruntime.InferenceSession,
    image: np.ndarray,
    score_th: Optional[float] = None,
) -> np.ndarray:
    # ONNX Input Size
    input_size = session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process: Resize, BGR->RGB, Normalize, Transpose, float32 cast
    input_image = cv2.resize(image, dsize=(input_width, input_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_image = (input_image / 255.0 - mean) / std  # type: ignore
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_image})

    # Post process: Squeeze, Sigmoid, Multiply by 255, uint8 cast
    mask = np.squeeze(result[-1])
    mask = sigmoid(mask)
    if score_th is not None:
        mask = np.where(mask < score_th, 0, 1)
    mask *= 255
    mask = mask.astype('uint8')

    return mask


if __name__ == "__main__":
    model_path = "./checkpoints/onnx/BiRefNet.onnx"

    # Load model
    providers=['CUDAExecutionProvider','CPUExecutionProvider']
    onnx_session = onnxruntime.InferenceSession( model_path, providers=providers)
    frame = Image.open("/home/leonel/heels01.jpg")

    mask = run_inference(onnx_session, np.array(frame))
    mask = cv2.resize(mask,dsize=frame.size)

    pred_pil = Image.fromarray(mask)
    image_masked = refine_foreground(frame, pred_pil, device="cpu")
    image_masked.putalpha(pred_pil)
    image_masked.save('result_onnx.png')






