//! Opening a web page in the user's browser.
//!
//! Every URL the launcher opens (the window's links and the Web UI after a start) goes through [`open_url`]. On Linux
//! it runs `xdg-open` as a child built by `spawn.rs`, so that under an AppImage `xdg-open` and the helpers it starts
//! get the launcher's own environment and not the `$APPDIR` library paths and GTK variables `AppRun` exported. macOS
//! and Windows use the system's own opener through the `open` crate.

use std::io;

/// Open `url` in the default browser without waiting for it.
///
/// # Errors
///
/// Fails when the system's opener cannot be started.
#[cfg(target_os = "linux")]
pub fn open_url(url: &str) -> io::Result<()> {
    let mut child = xdg_open(url).spawn()?;
    // `xdg-open` ends once it has handed the page over; reap it so that it leaves no zombie behind.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn open_url(url: &str) -> io::Result<()> {
    open::that_detached(url)
}

/// The `xdg-open <url>` command, with no input and no output.
#[cfg(target_os = "linux")]
fn xdg_open(url: &str) -> std::process::Command {
    use std::path::Path;
    use std::process::Stdio;

    let mut command = crate::spawn::hidden_command(Path::new("xdg-open"));
    command
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    command
}
