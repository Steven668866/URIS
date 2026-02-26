# Initialize Conda in PowerShell
# Run this script with: . .\init_conda.ps1

Write-Host "正在初始化 Conda 环境..." -ForegroundColor Green

# Add Conda to PATH for current session
$env:PATH += ";$env:USERPROFILE\Miniconda3;$env:USERPROFILE\Miniconda3\Scripts;$env:USERPROFILE\Miniconda3\Library\bin"

# Display Conda version
Write-Host "Conda 版本:" -ForegroundColor Cyan
conda --version

# List available environments
Write-Host "`n可用的 Conda 环境:" -ForegroundColor Cyan
conda env list

Write-Host "`n✅ Conda 初始化完成！现在可以使用 conda 命令了。" -ForegroundColor Green






