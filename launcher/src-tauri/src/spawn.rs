//! How the launcher starts child processes.
//!
//! Every child (nvidia-smi, uv, the tool checks, `ziv`) is built here: no shell, the program and its arguments stay
//! `OsString`s, and on Windows no console window opens.

use std::path::Path;
use std::process::Command;

/// `CREATE_NO_WINDOW`: a console program started without a console window. Never `DETACHED_PROCESS`.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Build a command for `program` that runs without a shell and, on Windows, without a window.
pub fn hidden_command(program: &Path) -> Command {
    no_window(Command::new(program))
}

#[cfg(windows)]
fn no_window(mut command: Command) -> Command {
    use std::os::windows::process::CommandExt;
    command.creation_flags(CREATE_NO_WINDOW);
    command
}

#[cfg(not(windows))]
fn no_window(command: Command) -> Command {
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_program_path_and_adds_no_arguments() {
        let program = Path::new("a folder").join("uv");
        let command = hidden_command(&program);
        assert_eq!(command.get_program(), program.as_os_str());
        assert_eq!(command.get_args().count(), 0);
    }
}
