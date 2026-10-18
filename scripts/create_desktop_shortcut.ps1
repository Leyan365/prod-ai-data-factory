# Script to create a Desktop shortcut for Training Data Factory Web Dashboard

$desktop = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$target = Join-Path -Path $PSScriptRoot -ChildPath "..\run_web.bat"
$target = [System.IO.Path]::GetFullPath($target)
$workDir = [System.IO.Path]::GetDirectoryName($target)

$wsh = New-Object -ComObject WScript.Shell
$sc = $wsh.CreateShortcut("$desktop\Training Data Factory.lnk")
$sc.TargetPath = $target
$sc.WorkingDirectory = $workDir
$sc.Description = "Launch Training Data Factory Modern Web Dashboard"
$sc.IconLocation = "shell32.dll,220"
$sc.Save()

Write-Host "Desktop shortcut created at: $desktop\Training Data Factory.lnk"
