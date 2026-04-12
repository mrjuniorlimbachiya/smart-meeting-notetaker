import subprocess
import os
import time

OUTPUT_PATH = r'C:\Users\limba\smart-meeting-notetaker\backend\recording.wav'

_ps_process = None
is_recording = False

POWERSHELL_RECORD = r"""
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class MCI {
    [DllImport("winmm.dll")]
    public static extern int mciSendString(string cmd, System.Text.StringBuilder ret, int len, IntPtr hwnd);
}
'@

[MCI]::mciSendString("open new Type waveaudio Alias recsound", $null, 0, [IntPtr]::Zero)
[MCI]::mciSendString("set recsound time format ms", $null, 0, [IntPtr]::Zero)
[MCI]::mciSendString("record recsound", $null, 0, [IntPtr]::Zero)

Write-Host "RECORDING_STARTED"
[Console]::Out.Flush()

# Wait for STOP signal via stdin
$line = [Console]::In.ReadLine()

[MCI]::mciSendString("stop recsound", $null, 0, [IntPtr]::Zero)
[MCI]::mciSendString("save recsound OUTPUT_FILE", $null, 0, [IntPtr]::Zero)
[MCI]::mciSendString("close recsound", $null, 0, [IntPtr]::Zero)

Write-Host "RECORDING_SAVED"
""".replace("OUTPUT_FILE", OUTPUT_PATH)

def start_recording():
    global _ps_process, is_recording
    is_recording = True

    _ps_process = subprocess.Popen(
        ['powershell', '-NoProfile', '-Command', POWERSHELL_RECORD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    # Wait for confirmation that recording started
    line = _ps_process.stdout.readline()
    print(f'Recording status: {line.strip()}')
    print('Recording started via Windows MCI')

def stop_recording():
    global _ps_process, is_recording
    is_recording = False

    if _ps_process and _ps_process.poll() is None:
        # Send STOP signal to the PowerShell process
        _ps_process.stdin.write("STOP\n")
        _ps_process.stdin.flush()

        # Wait for save confirmation
        line = _ps_process.stdout.readline()
        print(f'Save status: {line.strip()}')
        _ps_process.wait(timeout=10)

    # Verify file was saved
    if os.path.exists(OUTPUT_PATH):
        size = os.path.getsize(OUTPUT_PATH)
        print(f'Recording saved: {OUTPUT_PATH} ({size} bytes)')
    else:
        print(f'WARNING: recording.wav not found at {OUTPUT_PATH}')

    _ps_process = None
    print(f'Recording stopped')