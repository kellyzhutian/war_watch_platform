$ErrorActionPreference = "Stop"

$TargetFile = "$PSScriptRoot\start_desktop.ps1"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutFile = "$DesktopPath\War Watch Platform.lnk"

$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutFile)
$Shortcut.TargetPath = "cmd.exe"
$Shortcut.Arguments = "/k powershell -NoProfile -ExecutionPolicy Bypass -File `"$TargetFile`""
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.Description = "Launch War Watch Platform"
# Use a globe icon
$Shortcut.IconLocation = "shell32.dll,14" 
$Shortcut.Save()

Write-Host "Success: Shortcut created at $ShortcutFile"
