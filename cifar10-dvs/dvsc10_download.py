import argparse
import concurrent.futures
import hashlib
import os
import time
from pathlib import Path
from urllib.request import urlopen


RESOURCES = (
    (
        "airplane.zip",
        "https://ndownloader.figshare.com/files/7712788",
        "0afd5c4bf9ae06af762a77b180354fdd",
    ),
    (
        "automobile.zip",
        "https://ndownloader.figshare.com/files/7712791",
        "8438dfeba3bc970c94962d995b1b9bdd",
    ),
    (
        "bird.zip",
        "https://ndownloader.figshare.com/files/7712794",
        "a9c207c91c55b9dc2002dc21c684d785",
    ),
    (
        "cat.zip",
        "https://ndownloader.figshare.com/files/7712812",
        "52c63c677c2b15fa5146a8daf4d56687",
    ),
    (
        "deer.zip",
        "https://ndownloader.figshare.com/files/7712815",
        "b6bf21f6c04d21ba4e23fc3e36c8a4a3",
    ),
    (
        "dog.zip",
        "https://ndownloader.figshare.com/files/7712818",
        "f379ebdf6703d16e0a690782e62639c3",
    ),
    (
        "frog.zip",
        "https://ndownloader.figshare.com/files/7712842",
        "cad6ed91214b1c7388a5f6ee56d08803",
    ),
    (
        "horse.zip",
        "https://ndownloader.figshare.com/files/7712851",
        "e7cbbf77bec584ffbf913f00e682782a",
    ),
    (
        "ship.zip",
        "https://ndownloader.figshare.com/files/7712836",
        "41c7bd7d6b251be82557c6cce9a7d5c9",
    ),
    (
        "truck.zip",
        "https://ndownloader.figshare.com/files/7712839",
        "89f3922fd147d9aeff89e76a2b0b70a7",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download CIFAR10-DVS zip files only. "
            "No extraction or processing is performed."
        )
    )
    parser.add_argument(
        "--data-path",
        default="./data/CIFAR10DVS/download",
        help="Directory used to store the downloaded CIFAR10-DVS zip files.",
    )
    parser.add_argument(
        "--workers",
        default=10,
        type=int,
        help="Number of files to download concurrently.",
    )
    return parser.parse_args()


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024


def format_seconds(seconds):
    if seconds is None:
        return "--:--"

    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def build_progress_line(prefix, downloaded, total, start_time, state):
    if state:
        return f"{prefix} {state}"

    elapsed = max(time.time() - start_time, 1e-6)
    speed = downloaded / elapsed

    if total:
        percent = downloaded / total
        filled = int(percent * 28)
        bar = "#" * filled + "-" * (28 - filled)
        remaining = (total - downloaded) / speed if speed > 0 else None
        status = (
            f"{percent * 100:6.2f}% "
            f"{format_size(downloaded)}/{format_size(total)} "
            f"{format_size(speed)}/s ETA {format_seconds(remaining)}"
        )
    else:
        bar = "#" * 28
        status = f"{format_size(downloaded)} {format_size(speed)}/s ETA --:--"

    return f"{prefix} [{bar}] {status}"


def render_progress(progress):
    print("\033[H\033[J", end="")
    for item in progress:
        print(
            build_progress_line(
                item["prefix"],
                item["downloaded"],
                item["total"],
                item["start_time"],
                item["state"],
            )
        )


def download_file(url, target, progress_item):
    with urlopen(url) as response, target.open("wb") as f:
        progress_item["total"] = int(response.headers.get("Content-Length", 0))
        progress_item["downloaded"] = 0
        progress_item["start_time"] = time.time()
        progress_item["state"] = ""

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            progress_item["downloaded"] += len(chunk)


def download_resource(index, total_files, resource, data_path, progress_item):
    filename, url, expected_md5 = resource
    target = data_path / filename
    progress_item["prefix"] = f"[{index}/{total_files}] {filename}"

    if target.exists() and md5sum(target) == expected_md5:
        progress_item["state"] = f"done: existing verified file at {target}"
        return

    progress_item["state"] = "starting"
    download_file(url, target, progress_item)

    progress_item["state"] = "checking md5"
    actual_md5 = md5sum(target)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {target}: expected {expected_md5}, got {actual_md5}"
        )

    progress_item["state"] = "done"


def main():
    args = parse_args()
    data_path = Path("./data/CIFAR10DVS/download").expanduser()
    data_path.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(RESOURCES)))

    print(f"Downloading CIFAR10-DVS zip files to: {data_path}")
    total_files = len(RESOURCES)
    print(f"Concurrent downloads: {workers}")
    progress = [
        {
            "prefix": f"[{index}/{total_files}] {resource[0]}",
            "downloaded": 0,
            "total": 0,
            "start_time": time.time(),
            "state": "queued",
        }
        for index, resource in enumerate(RESOURCES, start=1)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_resource,
                index,
                total_files,
                resource,
                data_path,
                progress[index - 1],
            )
            for index, resource in enumerate(RESOURCES, start=1)
        ]

        try:
            while True:
                render_progress(progress)
                done = sum(future.done() for future in futures)
                if done == len(futures):
                    break
                time.sleep(0.5)
        finally:
            render_progress(progress)

        for future in futures:
            future.result()

    print("Done. Downloaded zip files only; no extraction or processing was performed.")


if __name__ == "__main__":
    main()
