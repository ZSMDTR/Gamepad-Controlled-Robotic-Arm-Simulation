param([string]$LeaderPort = '')

$project = Split-Path -Parent $PSScriptRoot
if ($project -match '[^\x00-\x7F]') {
    Write-Host 'MuJoCo may fail to load models from paths containing non-ASCII characters.'
    Write-Host 'Extract the project to an English-only path such as C:\arm-teleop, then run it there.'
    exit 1
}
Set-Location -LiteralPath $project
$venv = Join-Path $project '.venv'
$python = Join-Path $venv 'Scripts\python.exe'

if (!(Test-Path -LiteralPath $venv)) {
    Write-Host 'First run: locating Python 3.10 or newer...'
    $candidate = $null
    $candidateArgs = @()
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>$null
        if ($LASTEXITCODE -eq 0) { $candidate = $py.Source; $candidateArgs = @('-3') }
    }
    if (!$candidate) {
        $systemPython = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($systemPython) {
            & $systemPython.Source -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>$null
            if ($LASTEXITCODE -eq 0) { $candidate = $systemPython.Source }
        }
    }
    if (!$candidate) {
        Write-Host 'Python 3.10 or newer is required. Install it from https://www.python.org/downloads/windows/ and try again.'
        exit 1
    }
    Write-Host 'Creating the project virtual environment...'
    & $candidate @candidateArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Could not create .venv. Fix the Python installation, or remove the incomplete .venv yourself before retrying.'
        exit 1
    }
}

if (!(Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Host 'The existing .venv is incomplete. It was not changed. Move or remove it yourself, then retry.'
    exit 1
}
& $python -c 'import os,sys; sys.exit(not (sys.version_info >= (3,10) and os.path.normcase(os.path.realpath(sys.prefix)) == os.path.normcase(os.path.realpath(sys.argv[1]))))' $venv 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'The existing .venv is unusable, uses an older Python, or was moved from another location.'
    Write-Host 'It was not changed. Move or remove it yourself, then retry to create a new one.'
    exit 1
}

$check = @'
import importlib.metadata as m, re
def version(name):
    match = re.match(r'^(\d+)\.(\d+)', m.version(name))
    if not match: raise ValueError(name)
    return tuple(map(int, match.groups()))
assert (1, 24) <= version('numpy') < (3, 0)
assert (3, 2) <= version('mujoco') < (4, 0)
m.version('arm-teleop-sdls')
import numpy, mujoco, serial, arm_teleop, optimized_algorithm
'@
& $python -c $check 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing project dependencies. Internet access is needed on the first run...'
    & $python -m pip install -e $project
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Dependency installation failed. Check the network and Python installation, then retry.'
        exit 1
    }
}

$model = Join-Path $project 'robot_models\b601_rs'
if (!(Test-Path -LiteralPath $model)) {
    Write-Host 'Downloading and verifying the official B601-RS resources...'
    & $python (Join-Path $PSScriptRoot 'install_b601.py')
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Model download failed. Check the network and retry. Existing models are never overwritten.'
        exit 1
    }
} elseif (!(Test-Path -LiteralPath (Join-Path $model 'model.xml') -PathType Leaf) -or !(Test-Path -LiteralPath (Join-Path $model 'profile.json') -PathType Leaf) -or !(Test-Path -LiteralPath (Join-Path $model 'meshes') -PathType Container)) {
    Write-Host 'The existing B601-RS model directory is incomplete and was not changed.'
    Write-Host 'Move or remove it yourself, then retry to download the model.'
    exit 1
}

Write-Host 'Starting the B601-RS MuJoCo practice scene.'
Write-Host 'Keep this window open. Use the MuJoCo window to control the arm; close it to stop.'
if ($LeaderPort) {
    & $python (Join-Path $PSScriptRoot 'native_viewer.py') --robot b601_rs --leader-port $LeaderPort --leader-config (Join-Path $project 'leader-b601-rs.json')
} else {
    & $python (Join-Path $PSScriptRoot 'native_viewer.py') --robot b601_rs
}
exit $LASTEXITCODE
