$python = ".\.venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path "outputs\task24_TC136" | Out-Null

$args1 = @(
    "-m", "src.detection.retinaface",
    "--image", "data/test_images/TC-001.jpg",
    "--out", "outputs/task24_TC136/TC-136_A.jpg",
    "--device", "cpu"
)

$args2 = @(
    "-m", "src.detection.retinaface",
    "--image", "data/test_images/TC-003.jpg",
    "--out", "outputs/task24_TC136/TC-136_B.jpg",
    "--device", "cpu"
)

$p1 = Start-Process -FilePath $python -ArgumentList $args1 -PassThru
$p2 = Start-Process -FilePath $python -ArgumentList $args2 -PassThru

$p1.WaitForExit()
$p2.WaitForExit()

Write-Host "Process A exit code: $($p1.ExitCode)"
Write-Host "Process B exit code: $($p2.ExitCode)"

Write-Host "Output A exists: $(Test-Path 'outputs/task24_TC136/TC-136_A.jpg')"
Write-Host "Output B exists: $(Test-Path 'outputs/task24_TC136/TC-136_B.jpg')"
