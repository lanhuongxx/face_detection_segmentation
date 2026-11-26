$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$inputDir = Join-Path $PSScriptRoot "data\special_cases\TC-119_batch"
$outputDir = Join-Path $PSScriptRoot "outputs\task24_TC119"
$csvPath = Join-Path $outputDir "task24_batch_summary.csv"

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$images = Get-ChildItem $inputDir -File -Recurse |
    Where-Object { $_.Extension.ToLower() -in ".jpg", ".jpeg", ".png" } |
    Sort-Object Name

$results = @()

foreach ($img in $images) {
    $testCase = [System.IO.Path]::GetFileNameWithoutExtension($img.Name)
    $outPath = Join-Path $outputDir $img.Name

    Write-Host ""
    Write-Host "===== Dang chay $testCase =====" -ForegroundColor Cyan

    $start = Get-Date
    $stdout = & $python -m src.detection.retinaface --image $img.FullName --out $outPath --device cpu 2>&1
    $exitCode = $LASTEXITCODE
    $end = Get-Date

    $duration = [math]::Round(($end - $start).TotalSeconds, 2)
    $status = "Ran"
    $detectedFaces = ""
    $note = ($stdout | Out-String).Trim()

    if ($exitCode -ne 0) {
        $status = "Error"
    }
    else {
        $joined = ($stdout | Out-String)

        if ($joined -match "Đã phát hiện\s+(\d+)\s+mặt") {
            $detectedFaces = $matches[1]
        }
        elseif ($joined -match "Da phat hien\s+(\d+)\s+mat") {
            $detectedFaces = $matches[1]
        }
    }

    $results += [PSCustomObject]@{
        TestCase      = $testCase
        InputImage    = $img.FullName
        OutputImage   = $outPath
        Status        = $status
        DetectedFaces = $detectedFaces
        DurationSec   = $duration
        Note          = ($note -replace "`r?`n", " | ")
    }
}

$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Da chay xong toan bo anh." -ForegroundColor Green
Write-Host "Anh ket qua: $outputDir" -ForegroundColor Green
Write-Host "Bang tong hop: $csvPath" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green