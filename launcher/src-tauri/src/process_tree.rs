//! Stopping a server and everything it started, without touching other processes.
//!
//! The tree is snapshotted before anything is stopped, because a parent that dies orphans its children and the
//! parent links are gone afterwards. A snapshot member is only signalled while it is still alive with the start
//! time it had in the snapshot, so a PID the system reused for another program is left alone. Zombies count as
//! dead: the parent that started the process reaps them.
//!
//! [`descendants`] and [`same_process`] are pure; the rest reads the process table through `sysinfo`.

use std::collections::HashSet;
use std::thread;
use std::time::{Duration, Instant};

use sysinfo::{Pid, Process, ProcessRefreshKind, ProcessStatus, ProcessesToUpdate, Signal, System};

/// How long a server gets to exit after SIGTERM before its tree is killed.
pub const STOP_GRACE: Duration = Duration::from_secs(15);

/// How long to wait, after killing, for the members to disappear.
const KILL_WAIT: Duration = Duration::from_secs(3);

/// How often to look at the process table while waiting.
const POLL: Duration = Duration::from_millis(50);

/// How far apart (in seconds) two readings of one process's start time may be. The system derives a start time from
/// the boot time, which can shift by a second between readings.
const START_TIME_TOLERANCE_SECS: u64 = 5;

/// One row of the process table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub pid: u32,
    pub parent: Option<u32>,
    /// Seconds since the Unix epoch.
    pub start_time: u64,
}

/// A process in a snapshot: its id and the start time that proves it is still the same process later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Member {
    pub pid: u32,
    pub start_time: u64,
}

impl From<&Entry> for Member {
    fn from(entry: &Entry) -> Self {
        Self {
            pid: entry.pid,
            start_time: entry.start_time,
        }
    }
}

/// Whether two start times belong to the same process.
pub fn same_process(a: u64, b: u64) -> bool {
    a.abs_diff(b) <= START_TIME_TOLERANCE_SECS
}

/// Every process below `root` by parent links, nearest first; `root` itself is not included. A process that
/// points back up the tree (a reused PID) is visited once.
pub fn descendants(entries: &[Entry], root: u32) -> Vec<Member> {
    let mut seen = HashSet::from([root]);
    let mut found: Vec<Member> = Vec::new();
    let mut frontier = vec![root];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for entry in entries {
            if entry
                .parent
                .is_some_and(|parent| frontier.contains(&parent))
                && seen.insert(entry.pid)
            {
                found.push(Member::from(entry));
                next.push(entry.pid);
            }
        }
        frontier = next;
    }
    found
}

/// A handle on the process table; `None` when the system will not provide one, which reads as "nothing is running".
fn open_table() -> Option<System> {
    System::new().ok()
}

/// The process table now, without threads.
fn read_entries(system: &mut System) -> Vec<Entry> {
    system.refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::nothing());
    system
        .processes()
        .values()
        .filter(|process| process.thread_kind().is_none())
        .map(|process| Entry {
            pid: process.pid().as_u32(),
            parent: process.parent().map(Pid::as_u32),
            start_time: process.start_time(),
        })
        .collect()
}

/// `root` and everything below it, root first; empty when `root` is not running.
pub fn snapshot_tree(root: u32) -> Vec<Member> {
    let Some(mut system) = open_table() else {
        return Vec::new();
    };
    let entries = read_entries(&mut system);
    let Some(top) = entries.iter().find(|entry| entry.pid == root) else {
        return Vec::new();
    };
    let mut members = vec![Member::from(top)];
    members.extend(descendants(&entries, root));
    members
}

/// Whether `process` is the live process `member` was taken from.
fn is_member(process: &Process, member: &Member) -> bool {
    process.status() != ProcessStatus::Zombie
        && same_process(process.start_time(), member.start_time)
}

/// Re-read `members` and return the process handles of the ones still alive with their snapshot start time.
fn refresh_alive<'a>(system: &'a mut System, members: &[Member]) -> Vec<&'a Process> {
    let pids: Vec<Pid> = members
        .iter()
        .map(|member| Pid::from_u32(member.pid))
        .collect();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&pids),
        true,
        ProcessRefreshKind::nothing(),
    );
    members
        .iter()
        .filter_map(|member| {
            system
                .process(Pid::from_u32(member.pid))
                .filter(|process| is_member(process, member))
        })
        .collect()
}

/// The members of `members` that are still running.
#[cfg(test)]
pub fn alive_members(members: &[Member]) -> Vec<Member> {
    open_table().map_or_else(Vec::new, |mut system| alive_in(&mut system, members))
}

fn alive_in(system: &mut System, members: &[Member]) -> Vec<Member> {
    let alive: HashSet<u32> = refresh_alive(system, members)
        .iter()
        .map(|process| process.pid().as_u32())
        .collect();
    members
        .iter()
        .copied()
        .filter(|member| alive.contains(&member.pid))
        .collect()
}

/// The start time of the live process `pid`, or `None` when it is not running (or only a zombie).
pub fn start_time_of(pid: u32) -> Option<u64> {
    let mut system = open_table()?;
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[Pid::from_u32(pid)]),
        true,
        ProcessRefreshKind::nothing(),
    );
    system
        .process(Pid::from_u32(pid))
        .filter(|process| process.status() != ProcessStatus::Zombie)
        .map(Process::start_time)
}

/// Poll until none of `members` is alive or `limit` passes; returns the members still alive.
fn wait_gone(system: &mut System, members: &[Member], limit: Duration) -> Vec<Member> {
    let deadline = Instant::now() + limit;
    loop {
        let alive = alive_in(system, members);
        if alive.is_empty() || Instant::now() >= deadline {
            return alive;
        }
        thread::sleep(POLL);
    }
}

/// Stop every member of a snapshot. On Unix the first member (the server) gets SIGTERM and `grace` to take the
/// others down with it; survivors are then killed. On Windows every member is terminated at once. Returns the
/// members that could not be stopped (all of them when the system gives no process table); empty when the tree is
/// gone.
pub fn stop_members(members: &[Member], grace: Duration) -> Vec<Member> {
    let Some(root) = members.first() else {
        return Vec::new();
    };
    let Some(mut system) = open_table() else {
        return members.to_vec();
    };
    if cfg!(unix) {
        for process in refresh_alive(&mut system, std::slice::from_ref(root)) {
            process.kill_with(Signal::Term);
        }
        if wait_gone(&mut system, members, grace).is_empty() {
            return Vec::new();
        }
    }
    for process in refresh_alive(&mut system, members) {
        process.kill();
    }
    wait_gone(&mut system, members, KILL_WAIT)
}

/// Snapshot the tree below `root` (root included) and stop all of it; see [`stop_members`]. Nothing is stopped
/// unless `root` is alive with `root_start_time` (as [`same_process`] reads it), so a PID the system reused for
/// another program is left alone; nothing running at `root` is nothing to stop.
pub fn stop_tree(root: u32, root_start_time: u64, grace: Duration) -> Vec<Member> {
    let tree = snapshot_tree(root);
    match tree.first() {
        Some(top) if same_process(top.start_time, root_start_time) => stop_members(&tree, grace),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(pid: u32, parent: Option<u32>) -> Entry {
        Entry {
            pid,
            parent,
            start_time: u64::from(pid) * 10,
        }
    }

    fn pids(members: &[Member]) -> Vec<u32> {
        members.iter().map(|member| member.pid).collect()
    }

    #[test]
    fn descendants_follow_parent_links_through_every_level() {
        let table = [
            entry(1, None),
            entry(10, Some(1)),
            entry(11, Some(10)),
            entry(12, Some(11)),
            entry(13, Some(10)),
            entry(20, Some(1)),
        ];
        assert_eq!(pids(&descendants(&table, 10)), [11, 13, 12]);
        assert_eq!(pids(&descendants(&table, 1)), [10, 20, 11, 13, 12]);
    }

    #[test]
    fn descendants_carry_their_start_times() {
        let table = [entry(10, None), entry(11, Some(10))];
        assert_eq!(
            descendants(&table, 10),
            [Member {
                pid: 11,
                start_time: 110
            }]
        );
    }

    #[test]
    fn descendants_leave_unrelated_processes_out() {
        let table = [entry(10, None), entry(30, Some(2)), entry(31, Some(30))];
        assert!(descendants(&table, 10).is_empty());
        assert!(descendants(&table, 99).is_empty());
    }

    #[test]
    fn descendants_survive_a_cycle_and_a_self_parent() {
        let table = [
            entry(10, Some(12)),
            entry(11, Some(10)),
            entry(12, Some(11)),
            entry(13, Some(13)),
        ];
        assert_eq!(pids(&descendants(&table, 10)), [11, 12]);
    }

    #[test]
    fn start_times_match_within_five_seconds() {
        assert!(same_process(100, 100));
        assert!(same_process(100, 105));
        assert!(same_process(105, 100));
        assert!(!same_process(100, 106));
        assert!(!same_process(106, 100));
    }

    #[test]
    fn an_unused_pid_has_no_start_time_and_no_tree() {
        assert_eq!(start_time_of(u32::MAX - 1), None);
        assert!(snapshot_tree(u32::MAX - 1).is_empty());
        assert!(stop_tree(u32::MAX - 1, 0, Duration::ZERO).is_empty());
        assert!(stop_members(&[], Duration::ZERO).is_empty());
    }

    #[test]
    fn the_test_process_is_found_in_the_process_table() {
        let me = std::process::id();
        let start = start_time_of(me).expect("this process runs");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(start <= now + START_TIME_TOLERANCE_SECS, "{start} > {now}");
        let tree = snapshot_tree(me);
        assert_eq!(tree.first().map(|member| member.pid), Some(me));
    }

    #[cfg(unix)]
    mod unix {
        use std::process::{Child, Command, Stdio};

        use super::*;

        fn sh(script: &str) -> Child {
            Command::new("sh")
                .args(["-c", script])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap()
        }

        /// Wait until the tree under `root` has at least `count` members.
        fn tree_of(root: u32, count: usize) -> Vec<Member> {
            for _ in 0..200 {
                let tree = snapshot_tree(root);
                if tree.len() >= count {
                    return tree;
                }
                thread::sleep(Duration::from_millis(25));
            }
            panic!("the tree under {root} never reached {count} processes");
        }

        #[test]
        fn a_shell_with_background_children_is_stopped_entirely() {
            let mut child = sh("sleep 300 & sleep 301 & wait");
            let tree = tree_of(child.id(), 3);
            assert_eq!(tree[0].pid, child.id());
            let survivors = stop_members(&tree, Duration::from_millis(200));
            child.wait().unwrap();
            assert!(survivors.is_empty(), "{survivors:?}");
            assert!(alive_members(&tree).is_empty());
        }

        #[test]
        fn a_nested_tree_is_stopped_entirely() {
            let mut child = sh("sh -c 'sleep 302 & wait' & wait");
            let tree = tree_of(child.id(), 3);
            let survivors = stop_tree(child.id(), tree[0].start_time, Duration::from_millis(200));
            child.wait().unwrap();
            assert!(survivors.is_empty(), "{survivors:?}");
            assert!(alive_members(&tree).is_empty());
        }

        #[test]
        fn a_root_with_another_expected_start_time_is_left_alone() {
            let mut child = sh("sleep 305 & wait");
            let tree = tree_of(child.id(), 2);
            let wrong = tree[0].start_time + 1_000;
            let survivors = stop_tree(child.id(), wrong, Duration::from_millis(100));
            assert!(survivors.is_empty());
            assert!(child.try_wait().unwrap().is_none(), "the root was stopped");
            assert_eq!(
                alive_members(&tree).len(),
                tree.len(),
                "a member was stopped"
            );
            stop_tree(child.id(), tree[0].start_time, Duration::from_millis(200));
            child.wait().unwrap();
        }

        #[test]
        fn a_server_that_exits_on_sigterm_is_not_waited_out() {
            let mut child = sh("exec sleep 303");
            let tree = tree_of(child.id(), 1);
            let started = Instant::now();
            let survivors = stop_members(&tree, Duration::from_secs(30));
            child.wait().unwrap();
            assert!(survivors.is_empty());
            assert!(started.elapsed() < Duration::from_secs(10));
        }

        #[test]
        fn a_server_that_ignores_sigterm_is_killed_after_the_grace() {
            let mut child = sh("trap '' TERM; while :; do sleep 1; done");
            let tree = tree_of(child.id(), 1);
            thread::sleep(Duration::from_millis(200));
            let survivors = stop_members(&tree, Duration::from_millis(300));
            child.wait().unwrap();
            assert!(survivors.is_empty(), "{survivors:?}");
            assert!(alive_members(&tree).is_empty());
        }

        #[test]
        fn a_member_with_another_start_time_is_left_alone() {
            let mut child = sh("exec sleep 304");
            let real = tree_of(child.id(), 1)[0];
            let reused = Member {
                pid: real.pid,
                start_time: real.start_time + 1_000,
            };
            assert!(alive_members(&[reused]).is_empty());
            assert!(stop_members(&[reused], Duration::from_millis(100)).is_empty());
            assert!(
                child.try_wait().unwrap().is_none(),
                "the process was stopped"
            );
            child.kill().unwrap();
            child.wait().unwrap();
        }

        #[test]
        fn an_exited_child_is_not_alive_even_before_it_is_reaped() {
            let mut child = sh("sleep 0.5");
            let member = tree_of(child.id(), 1)[0];
            assert_eq!(alive_members(&[member]), [member]);
            for _ in 0..200 {
                if alive_members(&[member]).is_empty() {
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
            assert!(alive_members(&[member]).is_empty());
            child.wait().unwrap();
        }
    }
}
