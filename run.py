import os
import sys

# чтобы импорты transcriber.* работали при запуске из корня
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn

from transcriber.config.settings import settings


if __name__ == "__main__":
    uvicorn.run(
        "transcriber.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
