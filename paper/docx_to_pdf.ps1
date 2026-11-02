param([Parameter(Mandatory=$true)][string]$Docx, [string]$Pdf = "")
# Converts a .docx to PDF with Microsoft Word (COM automation). LibreOffice alternative:
#   soffice --headless --convert-to pdf --outdir paper paper/EV_Charging_Demand_Paper.docx
$Docx = (Resolve-Path $Docx).Path
if ($Pdf -eq "") { $Pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf") }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $true)
    $doc.Fields.Update() | Out-Null
    $doc.SaveAs([ref]$Pdf, [ref]17)   # 17 = wdFormatPDF
    $pages = $doc.ComputeStatistics(2)
    $doc.Close([ref]0)
    Write-Output "PDF: $Pdf ($pages pages)"
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
