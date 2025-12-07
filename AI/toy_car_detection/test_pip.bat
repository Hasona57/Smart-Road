@echo off
echo Testing pip upgrade...
py -m pip install --upgrade pip
echo Error level: %ERRORLEVEL%
if errorlevel 1 (
    echo FAILED
) else (
    echo SUCCESS
)
pause

