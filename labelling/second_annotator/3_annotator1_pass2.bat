@echo off
REM second session of the FIRST annotator (repeatability). Do not open labels_manual.json before or during this session.
call _run.bat eval_list labels_pass2.json 11
