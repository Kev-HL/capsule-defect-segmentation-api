"""
Measure end-to-end HTTP latency by POSTing an image to the /predict/ endpoint.

Usage (inside container):
  python /app/measure_http.py --url http://127.0.0.1:8000/predict/ --image static/samples/sample_0.png --requests 50 --warmup 5

Note 1: script file not included in image, copy it manually if needed.
Note 2: requests is commented out in the requirements.txt for building the image; uncomment it if needed or install manually (tested with requests==2.32.5).
"""

# IMPORTS
# Standard imports
import argparse
import statistics
import time
from pathlib import Path

# Third-party imports
# Note: requests is commented out in the requirements.txt for building the image; uncomment it if needed or install manually (tested with requests==2.32.5).
try:
    import requests
except ImportError:
    requests = None


# MAIN FUNCTION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000/predict/', help='Predict endpoint URL inside container (use 127.0.0.1)')
    parser.add_argument('--image', default='static/samples/sample_0.png', help='Image path inside the container')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    url = args.url
    image_path = Path(args.image)
    repeats = args.repeats
    warmup = args.warmup

    if not image_path.is_file():
        raise SystemExit(f'Image not found: {image_path}')

    if requests is None:
        raise SystemExit('requests not installed inside container')
    
    # Initialize session
    session = requests.Session()

    # Warmup
    for _ in range(warmup):
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            session.post(url, files=files, timeout=30)
            
    # Measurement
    times_ms = []
    for _ in range(repeats):
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/png')}
            t0 = time.perf_counter()
            r = session.post(url, files=files, timeout=30)
            t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        if r.status_code != 200:
            print('Non-200 response:', r.status_code, r.text)

    # Compute stats
    stats = {
        'count': len(times_ms),
        'mean_ms': statistics.mean(times_ms),
        'median_ms': statistics.median(times_ms),
        'stdev_ms': statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    }

    # Print results
    print('HTTP latency stats (ms):')
    for k, v in stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()