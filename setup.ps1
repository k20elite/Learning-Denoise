$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path venv)) { python -m venv venv }
& .\venv\Scripts\pip.exe install -r requirements.txt

Write-Host ".\venv\Scripts\python.exe denoise.py input.mp3"
