import os
import time

import psutil

from transcriber.services import metrics_db

INTERVAL = int(os.getenv("COLLECT_INTERVAL", "10"))      # сек между замерами
ROOTFS = os.getenv("HOST_ROOTFS", "/rootfs")             # хостовая ФС
DISK_PATHS = os.getenv("DISK_PATHS", "/,/srv,/var").split(",")
PRUNE_DAYS = int(os.getenv("PRUNE_DAYS", "30"))


def disk_stats():
    out = {}
    for p in DISK_PATHS:
        host_path = ROOTFS if p == "/" else ROOTFS + p
        try:
            u = psutil.disk_usage(host_path)
            out[p] = {"used": u.used, "total": u.total, "pct": u.percent}
        except Exception:
            pass
    return out


def sample():
    # cpu_percent с interval=1 сам спит 1с и даёт корректный % хоста
    cores = psutil.cpu_percent(interval=1, percpu=True)
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    try:
        l1, l5, l15 = os.getloadavg()
    except OSError:
        l1 = l5 = l15 = 0.0
    return {
        "ts": int(time.time()),
        "cpu": round(sum(cores) / len(cores), 1) if cores else 0.0,
        "cores": [round(c, 1) for c in cores],
        "mem_used": vm.used, "mem_total": vm.total, "mem_pct": vm.percent,
        "swap_used": sw.used, "swap_total": sw.total,
        "load1": round(l1, 2), "load5": round(l5, 2), "load15": round(l15, 2),
        "disks": disk_stats(),
    }


def main():
    metrics_db.init()
    print(f"[collector] старт: интервал {INTERVAL}с, БД {metrics_db.DB_PATH}", flush=True)
    last_prune = 0.0
    while True:
        try:
            metrics_db.insert_sample(sample())
        except Exception as error:
            print(f"[collector] ошибка замера: {error}", flush=True)
        now = time.time()
        if now - last_prune > 3600:
            try:
                metrics_db.prune(PRUNE_DAYS)
            except Exception as error:
                print(f"[collector] ошибка очистки: {error}", flush=True)
            last_prune = now
        # cpu_percent уже потратил ~1с
        time.sleep(max(1, INTERVAL - 1))


if __name__ == "__main__":
    main()
