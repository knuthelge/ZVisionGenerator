//! The Windows start-at-login value.
//!
//! The autostart plugin writes the `Run` value as the bare path of the launcher plus its arguments. An install
//! folder with a space in it (`C:\Users\Some One\AppData\Local\...`) then names a program that does not exist, so
//! Windows starts nothing. [`quote_run_value`] writes the value again with the path quoted.

#[cfg(any(windows, test))]
use std::path::Path;

/// The `Run` value for the launcher at `exe` with `arg`: the path in quotes, then the argument.
#[cfg(any(windows, test))]
fn run_command(exe: &Path, arg: &str) -> String {
    format!("\"{}\" {arg}", exe.display())
}

/// Write the current user's `Run` value `name` as the quoted path of this launcher followed by `arg`, replacing
/// what the autostart plugin wrote. `name` is the plugin's value name (the product name), which the Windows
/// uninstaller's hook deletes by.
///
/// # Errors
///
/// Fails when the launcher's own path is unknown or the registry cannot be written.
#[cfg(windows)]
pub fn quote_run_value(name: &str, arg: &str) -> Result<(), String> {
    const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
    let exe = std::env::current_exe().map_err(|error| error.to_string())?;
    windows_registry::CURRENT_USER
        .create(RUN_KEY)
        .and_then(|key| key.set_string(name, run_command(&exe, arg)))
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_path_is_quoted_and_the_argument_follows_it() {
        let exe =
            Path::new(r"C:\Users\Some One\AppData\Local\Z-Vision Generator\zvision-launcher.exe");
        assert_eq!(
            run_command(exe, "--autostart"),
            r#""C:\Users\Some One\AppData\Local\Z-Vision Generator\zvision-launcher.exe" --autostart"#
        );
    }
}
