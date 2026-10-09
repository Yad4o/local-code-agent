from agent import Config, ResourceGuard


def make(tmp_path, *, mem_kb=None, bats=(), temps=(), **cfg_over):
    meminfo = tmp_path / "meminfo"
    if mem_kb is not None:
        meminfo.write_text(f"MemTotal:  16000000 kB\nMemAvailable:  {mem_kb} kB\n")
    power = tmp_path / "power"
    power.mkdir(exist_ok=True)
    for i, (cap, status) in enumerate(bats):
        d = power / f"BAT{i}"
        d.mkdir(exist_ok=True)
        (d / "capacity").write_text(str(cap))
        (d / "status").write_text(status)
    (power / "AC").mkdir(exist_ok=True)                      # mains adapter dir must not count as a battery
    thermal = tmp_path / "thermal"
    thermal.mkdir(exist_ok=True)
    for i, t in enumerate(temps):
        d = thermal / f"thermal_zone{i}"
        d.mkdir(exist_ok=True)
        (d / "temp").write_text(str(t))
    cfg = Config(min_free_mem_mb=1500, min_battery_pct=20, max_temp_c=88.0, min_free_disk_mb=0)
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    return ResourceGuard(cfg, tmp_path, meminfo=meminfo, power_dir=power, thermal_dir=thermal)


# --- memory -------------------------------------------------------------------

def test_free_memory_is_parsed_in_mb(tmp_path):
    assert make(tmp_path, mem_kb=2048000).free_memory_mb() == 2000


def test_unreadable_memory_info_is_none_and_does_not_block(tmp_path):
    g = make(tmp_path)  # no meminfo file
    assert g.free_memory_mb() is None
    assert g.problems() == []


def test_low_memory_is_a_problem(tmp_path):
    probs = make(tmp_path, mem_kb=500_000).problems()
    assert len(probs) == 1 and "memory" in probs[0]


def test_enough_memory_is_fine(tmp_path):
    assert make(tmp_path, mem_kb=4_000_000).problems() == []


# --- battery ------------------------------------------------------------------

def test_low_battery_while_discharging_is_a_problem(tmp_path):
    probs = make(tmp_path, bats=[(10, "Discharging")]).problems()
    assert len(probs) == 1 and "battery" in probs[0]


def test_low_battery_on_charger_is_fine(tmp_path):
    assert make(tmp_path, bats=[(10, "Charging")]).problems() == []
    assert make(tmp_path, bats=[(10, "Full")]).problems() == []
    assert make(tmp_path, bats=[(10, "Not charging")]).problems() == []


def test_healthy_battery_discharging_is_fine(tmp_path):
    assert make(tmp_path, bats=[(80, "Discharging")]).problems() == []


def test_no_battery_is_fine(tmp_path):
    assert make(tmp_path).problems() == []


# --- temperature --------------------------------------------------------------

def test_hot_cpu_is_a_problem(tmp_path):
    probs = make(tmp_path, temps=[60000, 92000]).problems()
    assert len(probs) == 1 and "temperature" in probs[0]


def test_cool_cpu_is_fine_and_max_zone_is_used(tmp_path):
    g = make(tmp_path, temps=[40000, 71000, 55000])
    assert g.max_temp_c() == 71.0
    assert g.problems() == []


def test_garbage_temperatures_are_ignored(tmp_path):
    g = make(tmp_path, temps=["n/a", -274000, 999999999, 50000])
    assert g.max_temp_c() == 50.0


def test_no_sensors_is_fine(tmp_path):
    g = make(tmp_path)
    assert g.max_temp_c() is None
    assert g.problems() == []


# --- disk ---------------------------------------------------------------------

def test_low_disk_is_a_problem(tmp_path):
    probs = make(tmp_path, min_free_disk_mb=10_000_000).problems()
    assert len(probs) == 1 and "disk" in probs[0]


# --- combined + waiting -------------------------------------------------------

def test_multiple_problems_are_all_reported(tmp_path):
    probs = make(tmp_path, mem_kb=100_000, temps=[95000], bats=[(5, "Discharging")]).problems()
    assert len(probs) == 3


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_wait_returns_immediately_when_ok(tmp_path):
    c = Clock()
    g = make(tmp_path, mem_kb=4_000_000)
    assert g.wait_until_ok(sleep=c.sleep, now=c.now) is True
    assert c.t == 0


def test_wait_polls_until_the_problem_clears(tmp_path):
    c = Clock()
    g = make(tmp_path, mem_kb=100_000, resource_wait_s=600)
    seen = []

    def clear_after_three_polls(reasons):
        seen.append(reasons)
        if len(seen) == 3:
            (tmp_path / "meminfo").write_text("MemAvailable: 4000000 kB\n")

    assert g.wait_until_ok(sleep=c.sleep, now=c.now, on_wait=clear_after_three_polls) is True
    assert len(seen) == 3 and "memory" in seen[0][0]


def test_wait_gives_up_after_the_limit(tmp_path):
    c = Clock()
    g = make(tmp_path, mem_kb=100_000, resource_wait_s=60)
    assert g.wait_until_ok(sleep=c.sleep, now=c.now) is False
    assert 60 <= c.t <= 70


# --- stop requests are honoured while waiting -----------------------------------

def test_wait_stops_immediately_when_a_stop_is_requested(tmp_path):
    c = Clock()
    g = make(tmp_path, mem_kb=100_000, resource_wait_s=600)
    assert g.wait_until_ok(sleep=c.sleep, now=c.now, should_stop=lambda: True) is False
    assert c.t == 0


def test_wait_notices_a_stop_requested_mid_wait_within_a_second_or_two(tmp_path):
    c = Clock()
    g = make(tmp_path, mem_kb=100_000, resource_wait_s=600)
    assert g.wait_until_ok(sleep=c.sleep, now=c.now, should_stop=lambda: c.t >= 2) is False
    assert c.t <= 3                                  # not the whole 600s (or even a whole 5s poll)
