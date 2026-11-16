@echo off
REM usage: _run.bat <list: practice_list|eval_list> <output name> <shuffle seed>
if "%PY%"=="" set PY=python
for %%D in (vera scut) do (
  echo.
  echo ===== %%D : %1 =====
  %PY% tool\label_tool.py --ds_dir data\%%D --names_file data\%%D\%1.json --labels_name %2 --shuffle_seed %3 --max_h 700
)
echo.
echo Finished. Labels are in data\vera\%2 and data\scut\%2
pause
