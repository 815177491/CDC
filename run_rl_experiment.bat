@echo off
REM 柴油机RL对比实验快速启动脚本
REM 使用方法: run_rl_experiment.bat [--quick|--full-only] [--episodes 100] [--steps 200]

setlocal enabledelayedexpansion

REM 获取脚本所在目录
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM 运行Python脚本，传递所有参数
python run_gpu_comparison.py %*

pause
