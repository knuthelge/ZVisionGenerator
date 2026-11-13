; NSIS installer hooks (bundle.windows.nsis.installerHooks).
;
; Before the files are removed, run the launcher headless so it stops the server it started and the other launcher
; processes. Whether the data root and the start-at-login registry values go too depends on why the uninstaller runs:
;   - A real uninstall (Settings > Apps, or Uninstall in the app) starts uninstall.exe, which copies itself to %TEMP%
;     and runs from there, so $EXEDIR is not $INSTDIR. It deletes the root and the registry values.
;   - A reinstall over an existing install ("Uninstall before installing") runs the old uninstall.exe in place
;     (`_?=<install dir>`), so $EXEDIR is $INSTDIR; the updater passes /UPDATE ($UpdateMode = 1). Both only stop
;     things (`--keep-data`): the data and the start-at-login setting belong to the install that comes next.
;     Tauri's own uninstaller section deletes the Run value unless $UpdateMode = 1, which an in-place reinstall is not,
;     so the hook saves the value here and writes it back in NSIS_HOOK_POSTUNINSTALL, which runs after that delete in
;     the same process. The StartupApproved value is left alone, because nothing deletes it on this path.
; The value name is the app name tauri-plugin-autostart writes, which is the product name.

; Declared at file scope, like the template's own uninstaller variables; empty unless an in-place uninstall saved it.
Var SavedAutostartCommand

!macro NSIS_HOOK_PREUNINSTALL
  ${If} $EXEDIR != $INSTDIR
  ${AndIf} $UpdateMode <> 1
    ExecWait '"$INSTDIR\${MAINBINARYNAME}.exe" --uninstall-cleanup'
    DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${PRODUCTNAME}"
    DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run" "${PRODUCTNAME}"
  ${Else}
    ExecWait '"$INSTDIR\${MAINBINARYNAME}.exe" --uninstall-cleanup --keep-data'
    ReadRegStr $SavedAutostartCommand HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${PRODUCTNAME}"
  ${EndIf}
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $SavedAutostartCommand != ""
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${PRODUCTNAME}" "$SavedAutostartCommand"
  ${EndIf}
!macroend
