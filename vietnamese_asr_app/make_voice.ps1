Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("thanh_voice.wav")
$synth.Speak("Xin chào tôi tên là Thanh")
$synth.Dispose()
Write-Host "Audio generated successfully!"
Get-Item thanh_voice.wav
