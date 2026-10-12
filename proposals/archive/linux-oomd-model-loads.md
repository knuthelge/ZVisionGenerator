# Linux: keep systemd-oomd from killing the editor during model loads

**Status:** Accepted (2026-10-10). Applied as a local drop-in with 80% and 60 s; see [Results with the drop-in](#results-with-the-drop-in). The docs note for users is still to do.

## Problem

On Ubuntu, loading a large model can make `systemd-oomd` kill the terminal or editor the job was started from, even when the load would have finished. The job dies with it.

`systemd-oomd` does not watch for memory exhaustion here. It watches PSI memory pressure: the share of time tasks stall waiting for memory reclaim. Ubuntu's defaults kill a cgroup when that pressure stays above 50% for 20 s. Loading 15-35 GB of safetensors weights on a 32 GB-class machine fills the page cache and forces sustained reclaim, which can hold pressure above that line for the whole load.

oomd kills a whole cgroup, not a process: the descendant of `user@<uid>.service` with the most reclaim activity. A job started from the editor's integrated terminal lives in the editor's scope, so the editor dies with it. A job started from a standalone terminal takes that terminal's scope.

## Evidence

Test machine: Ubuntu 26.04 LTS with systemd 259, a 10 GB GPU, 32 GB-class RAM and a small swap file. Checked 2026-10-09 and 2026-10-10.

### Defaults in force

| Setting | Value | Source |
|---|---|---|
| `ManagedOOMMemoryPressure` | `kill` | `/usr/lib/systemd/system/user@.service.d/10-oomd-user-service-defaults.conf` |
| `ManagedOOMMemoryPressureLimit` | `50%` | same file (upstream's default is 60%, so Ubuntu is stricter) |
| `DefaultMemoryPressureDurationSec` | `20s` | `/usr/lib/systemd/oomd.conf.d/10-oomd-defaults.conf` |
| `ManagedOOMSwap` | `auto` on the root slice | oomd's swap limit is 90% |

### Kills during model testing

From `journalctl -u systemd-oomd`: four kills during one day of model testing, at 66-90% pressure against the 50% limit. Each entry reads "for > 20s with reclaim activity". Three hit the editor's scope and one a terminal's scope (`vte-spawn-*.scope`).

The two kills at 66-69% are the case this proposal is about: moderate pressure, only just over the limit. The other two, at about 80% and 90%, are higher and might have been real thrashing. The journal doesn't say whether those loads would have finished.

### Load sizes

[Quantization parity on CUDA](cuda-quant-parity.md) records the memory behind these loads: bf16 Krea 2 streams about 34.5 GB of weights, and building a q8 copy took about 25.6 GB before it was changed to stream. Both exceed what the page cache can hold on a 32 GB-class machine.

### What systemd offers

- `ManagedOOMMemoryPressureDurationSec=` is a per-unit override in systemd 257 and later. It is documented in `systemd.resource-control(5)`.
- `ManagedOOMPreference=avoid|omit` only takes effect for cgroups owned by root. A user unit can't use it to shield the editor.

### Fallback in use

Dev verification runs each heavy job as its own transient user unit, outside the editor's scope:

```bash
systemd-run --user --unit=ziv-verify-<job> --collect -p OOMScoreAdjust=500 ...
```

There is no `MemoryMax` cap. Then oomd's victim is the job, not the editor. That isn't guaranteed, because oomd picks by reclaim activity, and the job still dies on a false alarm.

### Measured on 2026-10-10 (before the drop-in)

Pressure and kill events from separate runs on the same machine:

| Load | Result |
|---|---|
| `zit` bf16 (25.5 GB peak RSS), in its own systemd unit | oomd killed the job at 65% pressure. The editor survived. |
| Krea 2 bf16 | oomd killed the editor's scope first (75.9%), then the job 15 s later (89.5%). |
| `klein4b` jobs (unit memory peaks of 11 to 20 GB, including page cache) | PSI `some avg10` stayed at or below 1.1, except one failed LoRA load at 19.1. |

A separate unit isn't enough for the biggest loads. The editor can still be the first victim, which strengthens the case for the drop-in.

### Not measured

- bf16 Krea 2 under the new limit. It is the biggest load (about 34 GB) and was not part of the end-to-end run, so it remains untested.
- Fedora. Its `systemd-oomd` defaults have not been checked.

## Results with the drop-in

The drop-in (80%, 60 s) is installed, and `oomctl` shows "Memory Pressure Limit: 80.00%, Duration: 1min". An end-to-end run followed: 18 jobs across `klein4b`, `zit`, `krea2` and `klein9b` (bf16 Krea 2 not run). Pressure was sampled at 1 Hz as PSI `some avg10` of the user service.

| Job | Samples (s) | Seconds above 50% | Seconds above 80% | Max avg10 | Killed |
|---|---|---|---|---|---|
| zit bf16 | 247 | 82 | 28 | 89.8 | no |
| zit bf16 + 2x upscale | 274 | 91 | 47 | 91.4 | no |
| klein9b q8 + LoRA | 253 | 35 | 5 | 84.2 | no |
| zit q4 (first use, builds stored copy) | 255 | 14 | 0 | 68.7 | no |
| krea2 q4 (first use) | 299 | 0 | 0 | 41.2 | no |

- oomd killed nothing in the whole 18-job run.
- With the default 50% for 20 s, both `zit` bf16 jobs would very likely have been killed: they spent 82-91 s above 50%. The samples don't show whether that time was contiguous, hence "very likely".
- At 80% for 60 s, no job stayed above 80% for a full minute. The worst cumulative time was 47 s (`zit` bf16 with the upscale).

Conclusion: 80% and 60 s held for this run and removed the false kills. The margin is not large, though. The heaviest job spent 47 s above 80% in total, with a peak of 91.4%, so a slightly longer or heavier load could cross the 60 s window if the pressure were contiguous. bf16 Krea 2 is heavier still and remains untested under the new limit.

## Proposed change

### 1. A drop-in that relaxes oomd for the user manager

The user installs it with sudo. The app never does.

```ini
# /etc/systemd/system/user@.service.d/99-oomd-relax.conf
[Service]
ManagedOOMMemoryPressureLimit=80%
ManagedOOMMemoryPressureDurationSec=60s
```

- **Apply:** `sudo systemctl daemon-reload`, then log out and in, or reboot. The drop-in only applies when the user manager restarts, which also restarts the editor.
- **Verify:** `oomctl` shows the 80% limit under "Memory Pressure Monitored CGroups". `systemctl show user@$(id -u).service -p ManagedOOMMemoryPressureLimit` prints it too.
- **Revert:** delete the file and repeat the reload and re-login.

### 2. Why these values

80% for 60 s still catches real thrashing. The two kills at about 80% and 90% were higher, and a longer window judges them more carefully. It tolerates the bursty reclaim of a weight load, which the two kills at 66-69% look like.

The thresholds were first a judgement call. [Results with the drop-in](#results-with-the-drop-in) measured them over a full run; they held, with a modest margin on the heaviest job. To record pressure during a load:

```bash
cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/memory.pressure
```

Sample `avg10` and `avg60` every few seconds for the whole load. Compare the peak against 80% and the time spent above 50% and 80%. Tune the limit and duration to sit above what a healthy load reaches and below what a real out-of-memory case reaches.

### 3. Product side

Later, and always optional. The app never changes system configuration.

- **Docs.** Add a note for Linux users, titled along the lines of "systemd-oomd closes my terminal or editor during model loads". It explains the cause, shows the drop-in and how to apply, verify and revert it, and describes the separate-unit practice. It could go in `docs/getting-started.md` or a Linux troubleshooting page (see open questions). Add a `CHANGELOG.md` entry under `[Unreleased]` when the docs change ships.
- **Possible follow-up.** `ziv-ui` on Linux could read `ManagedOOMMemoryPressureLimit` through `systemctl show` and warn once when it is the stricter default, pointing to the docs note. Skip it where `systemctl` is missing or the user manager isn't systemd.

### 4. Keep the separate-unit practice

It is complementary. The drop-in decides whether anything gets killed. The separate unit decides who gets killed when something does.

## Alternatives considered

- **Keep the defaults and use separate units only.** Cheap and needs no sudo, but each false alarm costs a rerun, and the victim isn't guaranteed to be the job.
- **`ManagedOOMMemoryPressure=auto` for the user manager**, which disables pressure kills. Rejected: with only a small swap file, a real out-of-memory case thrashes the desktop for minutes before the kernel OOM killer acts.
- **Disable `systemd-oomd`.** Rejected for the same reason. It also drops swap protection.
- **`MemoryMax` caps on jobs.** Rejected: caps make large loads fail outright, and the user doesn't want them.
- **`ManagedOOMPreference=avoid` on the editor.** Not possible: it only takes effect for cgroups owned by root.

## Open questions

- Whether 80% and 60 s keep enough margin for bf16 Krea 2 and for heavier chains such as an upscale after a bf16 generation. Measure it the same way.
- Whether other distros ship the same defaults. Fedora has enabled `systemd-oomd` by default since 34. Check the current values on each before writing user docs.
- Whether the docs note belongs in `docs/getting-started.md` or a separate Linux troubleshooting page.
