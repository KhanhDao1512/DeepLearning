import json
import os
import sys

class Tee:
    '''Ghi mọi thứ in ra màn hình (stdout) đồng thời vào file log.'''
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def start_logging(name: str):
    sys.stdout = Tee(f"logs/{name}.log")

def save_results(name: str, results: dict):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
