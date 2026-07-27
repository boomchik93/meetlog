import json
import sys

import numpy as np


def main():
    npy_path = sys.argv[1]
    model_path = sys.argv[2]

    audio = np.load(npy_path).astype(np.float32)

    # импорт только здесь: libggml из pywhispercpp конфликтует с llama.cpp,
    # поэтому whisper.cpp живёт в отдельном процессе
    from pywhispercpp.model import Model
    model = Model(model_path, print_progress=False, redirect_whispercpp_logs_to=sys.stderr)
    segments = model.transcribe(audio, language="ru")

    out = [{"t0": s.t0, "t1": s.t1, "text": s.text} for s in segments]
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
