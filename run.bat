@echo off
REM 使用项目Conda环境运行Python脚本
D:\Anaconda\Scripts\conda.exe run -p "%~dp0\.conda" --no-capture-output python %*
