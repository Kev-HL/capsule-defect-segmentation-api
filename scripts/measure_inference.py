"""
Measure TFLite interpreter inference latency and peak memory (RSS).

Usage (inside container):
  python /app/measure_inference.py --image static/samples/sample_0.png --model ./final_model.tflite --num_threads 2 --repeats 200 --warmup 20

Note 1: script file not included in image, copy it manually if needed.
Note 2: psutil is commented out in the requirements.txt for building the image; uncomment it if needed or install manually (tested with psutil==7.1.3).
"""

# IMPORTS
# Standard imports
import argparse
import time
import threading
import statistics
from pathlib import Path

# Third-party imports
import numpy as np
# Note: psutil is commented out in the requirements.txt for building the image; uncomment it if needed or install manually (tested with psutil==7.1.3).
try:
    import psutil
except ImportError:
    psutil = None
from PIL import Image
from tflite_runtime.interpreter import Interpreter
# from ai_edge_litert.interpreter import Interpreter

# AUXILIARY FUNCTIONS

# Function to sample memory usage in a separate thread
def sample_memory(pid: int, interval: float, stop_event: threading.Event, samples: list) -> None:
    proc = psutil.Process(pid)
    while not stop_event.is_set():
        try:
            samples.append(proc.memory_info().rss)
        except Exception:
            pass
        time.sleep(interval)

# Function to load and preprocess an image for inference
def load_and_preprocess_image(image_path: str, size=(512,512)) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, resample=Image.Resampling.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

# Function to measure inference latency and memory usage
def measure_inference(interpreter: Interpreter, input_array: np.ndarray, repeats=200, warmup=20, mem_sample_interval=0.01) -> dict:
    input_details = interpreter.get_input_details()
    input_index = input_details[0]['index']

    # Ensure input dtype matches expected dtype
    expected_dtype = input_details[0]['dtype']
    if input_array.dtype != expected_dtype:
        input_array = input_array.astype(expected_dtype)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_index, input_array)
        interpreter.invoke()

    # Memory sampling
    samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(psutil.Process().pid, mem_sample_interval, stop_event, samples), daemon=True)
    sampler.start()

    # Timed runs
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        interpreter.set_tensor(input_index, input_array)
        interpreter.invoke()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)  # ms

    # Stop memory sampling and wait for thread to finish
    stop_event.set()
    sampler.join(timeout=1.0)

    stats = {
        'count': len(timings),
        'mean_ms': statistics.mean(timings),
        'median_ms': statistics.median(timings),
        'stdev_ms': statistics.stdev(timings) if len(timings) > 1 else 0.0,
        'p95_ms': statistics.quantiles(timings, n=100)[94] if len(timings) >= 100 else None,
        'p99_ms': statistics.quantiles(timings, n=100)[98] if len(timings) >= 100 else None,
        'peak_rss_bytes': max(samples) if samples else None,
        'baseline_rss_bytes': samples[0] if samples else None,
    }
    return stats


# MAIN FUNCTION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./final_model.tflite', help='Path to tflite model inside the container')
    parser.add_argument('--image', default='static/samples/sample_0.png', help='Path to a sample PNG inside the container')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads for TFLite interpreter')
    parser.add_argument('--repeats', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    args = parser.parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    num_threads = args.num_threads
    repeats = args.repeats
    warmup = args.warmup

    if not model_path.is_file():
        raise SystemExit(f'Model not found at {model_path}')

    if not image_path.is_file():
        raise SystemExit(f'Image not found at {image_path}')
    
    if psutil is None:
        raise SystemExit('psutil is required for memory measurement; please install it in the environment.')
    
    input_array = load_and_preprocess_image(str(image_path), size=(512,512))

    if num_threads > 1:
        try:
            interpreter = Interpreter(model_path=str(model_path), num_threads=num_threads)
        except:
            print(f'Warning: num_threads={num_threads} not supported, falling back to single-threaded interpreter.')
            interpreter = Interpreter(model_path=str(model_path))
    else:
        interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    stats = measure_inference(interpreter, input_array, repeats=repeats, warmup=warmup)
    print('Inference stats (ms):')
    for k,v in stats.items():
        if v is None:
            print(f'  {k}: None')
        else:
            if k.endswith('_bytes'):
                print(f'  {k}: {v} bytes ({v/1024/1024:.2f} MB)')
            else:
                print(f'  {k}: {v}')


if __name__ == '__main__':
    main()