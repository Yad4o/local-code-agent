"""Resource guard: pause the agent instead of crashing the laptop.

Checks free RAM, battery (only when discharging), CPU temperature and free
disk. Every source is optional: if a sensor can't be read it simply does not
block (a missing sensor must never stop a 24/7 agent forever).
"""
import shutil
import time
from pathlib import Path
from typing import Callable

from codeagent.config import Config

POLL_SECONDS = 5.0


class ResourceGuard:
    def __init__(self, cfg: Config, workdir: str | Path, *,
                 meminfo: str | Path = "/proc/meminfo",
                 power_dir: str | Path = "/sys/class/power_supply",
                 thermal_dir: str | Path = "/sys/class/thermal"):
        self.cfg = cfg
        self.workdir = Path(workdir)
        self.meminfo = Path(meminfo)
        self.power_dir = Path(power_dir)
        self.thermal_dir = Path(thermal_dir)

    # --- readings ------------------------------------------------------------
    def free_memory_mb(self) -> int | None:
        try:
            for line in self.meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        return None

    def battery(self) -> tuple[int, bool] | None:
        """(percent, discharging) of the first battery, or None if there is none."""
        try:
            for d in sorted(self.power_dir.glob("BAT*")):
                cap = int((d / "capacity").read_text().strip())
                status = (d / "status").read_text().strip().lower()
                return cap, status == "discharging"
        except (OSError, ValueError):
            pass
        return None

    def max_temp_c(self) -> float | None:
        temps = []
        try:
            for f in self.thermal_dir.glob("thermal_zone*/temp"):
                try:
                    c = int(f.read_text().strip()) / 1000
                except (OSError, ValueError):
                    continue
                if 1 <= c <= 150:                 # ignore garbage readings
                    temps.append(c)
        except OSError:
            pass
        return max(temps) if temps else None

    def free_disk_mb(self) -> int | None:
        try:
            return shutil.disk_usage(self.workdir).free // (1024 * 1024)
        except OSError:
            return None

    # --- decision --------------------------------------------------------------
    def problems(self) -> list[str]:
        out = []
        mem = self.free_memory_mb()
        if mem is not None and mem < self.cfg.min_free_mem_mb:
            out.append(f"low memory: {mem} MB free (need {self.cfg.min_free_mem_mb} MB)")
        bat = self.battery()
        if bat and bat[1] and bat[0] < self.cfg.min_battery_pct:
            out.append(f"low battery: {bat[0]}% and discharging (need {self.cfg.min_battery_pct}%)")
        temp = self.max_temp_c()
        if temp is not None and temp > self.cfg.max_temp_c:
            out.append(f"high temperature: {temp:.0f}°C (limit {self.cfg.max_temp_c:.0f}°C)")
        disk = self.free_disk_mb()
        if disk is not None and disk < self.cfg.min_free_disk_mb:
            out.append(f"low disk space: {disk} MB free (need {self.cfg.min_free_disk_mb} MB)")
        return out

    def wait_until_ok(self, *, sleep: Callable[[float], None] = time.sleep,
                      now: Callable[[], float] = time.monotonic,
                      on_wait: Callable[[list[str]], None] | None = None,
                      should_stop: Callable[[], bool] | None = None) -> bool:
        """Block until resources are fine. False if they stayed bad past resource_wait_s,
        or if a stop was requested (checked every second, so Ctrl+C is never ignored)."""
        deadline = now() + self.cfg.resource_wait_s
        while True:
            if should_stop and should_stop():
                return False
            reasons = self.problems()
            if not reasons:
                return True
            if on_wait:
                on_wait(reasons)
            if now() >= deadline:
                return False
            for _ in range(int(POLL_SECONDS)):          # sleep in 1s slices
                if should_stop and should_stop():
                    return False
                sleep(1.0)
