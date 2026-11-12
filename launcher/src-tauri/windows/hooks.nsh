; NSIS installer hooks (bundle.windows.nsis.installerHooks).
;
; Before the files are removed, run the launcher headless so it stops the server it started and deletes its data root,
; then remove the start-at-login registry values. The value name is the app name tauri-plugin-autostart writes,
; which is the product name.
!macro NSIS_HOOK_PREUNINSTALL
  ExecWait '"$INSTDIR\${MAINBINARYNAME}.exe" --uninstall-cleanup'
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${PRODUCTNAME}"
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run" "${PRODUCTNAME}"
!macroend
