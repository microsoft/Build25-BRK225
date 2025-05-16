# Create the model folder
$targetPath = Join-Path $PSScriptRoot 'Model'
if (-not (Test-Path $targetPath -PathType Container)) {
    New-Item -Path $targetPath -ItemType Directory | Out-Null
}

# Download the model file
$modelPath = Join-Path $targetPath 'SqueezeNet.onnx'
if (-not (Test-Path $modelPath -PathType Leaf)) {
    Invoke-WebRequest `
        -Uri 'https://github.com/microsoft/Windows-Machine-Learning/blob/02b586811c8beb1ae2208c8605393267051257ae/SharedContent/models/SqueezeNet.onnx?raw=true' `
        -OutFile $modelPath
}

# Download the labels file
$labelsPath = Join-Path $targetPath 'SqueezeNet.Labels.txt'
if (-not (Test-Path $labelsPath -PathType Leaf)) {
    Invoke-WebRequest `
        -Uri 'https://github.com/microsoft/Windows-Machine-Learning/blob/02b586811c8beb1ae2208c8605393267051257ae/Samples/SqueezeNetObjectDetection/Desktop/cpp/Labels.txt?raw=true' `
        -OutFile $labelsPath
}
