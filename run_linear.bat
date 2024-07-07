@echo off
setlocal enabledelayedexpansion

for %%i in (0 1) do (
    for %%g in (0.4 0.6 0.8 0.99 1.0) do (
        for %%c in (0.0001 0.01 1 100 10000) do (
            python linear.py ^
            --cov %%c ^
            --gamma %%g ^
            --isgood %%i
        )
    )
)

pause