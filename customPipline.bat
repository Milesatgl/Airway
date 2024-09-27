@echo off

SETLOCAL

set start=%time%

set PYTHON_EXE="D:\mambaforge\envs\skeletonA\python.exe"
set BLENDER_EXE="D:\blender338\blender.exe"
set SCRIPT_DIR="E:\conda_projects\Airway\custom"
set projDir=%1
@REM set voxelScale=%2
set logDir=%projDir%\logs
mkdir %logDir%
@REM set stlPath=%projDir%\Airway.stl

@rem preprocess
echo STEP 0: preprocessing
%PYTHON_EXE% %SCRIPT_DIR%\preprocessing.py %projDir% >> %logDir%\preprocessing.log 2>&1

if errorlevel 0 (
    echo Preprocessing succeeded
    goto BFS
) else (
    echo Preprocessing failed with exitCode %ERRORLEVEL%, terminating script
    goto end
)

:BFS
echo STEP 1: BFS distance analysis
%PYTHON_EXE% %SCRIPT_DIR%\bfsDistance.py %projDir% >> %logDir%\bfsDistance.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Distance analysis succeeded
    goto PauseBat
) else (
    echo Distance analysis failed with exitCode %ERRORLEVEL%, terminating script
    goto end
)

:PauseBat
echo.
echo Please verify skeleton structure first. Press Y to continue, N to exit.
choice /M "Y/N"
if ERRORLEVEL 2 (
    echo User exit
    goto end
) else (
    echo User continue
    goto CreateTree
)

:CreateTree
echo STEP 2: Creating tree
%PYTHON_EXE% %SCRIPT_DIR%\createTree.py %projDir% >> %logDir%\createTree.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Creating tree succeeded
    goto ComposeTree
) else (
    echo Creating tree failed with exitCode %ERRORLEVEL%, terminating script
    goto end
)

:ComposeTree
echo STEP 3: Composing tree
%PYTHON_EXE% %SCRIPT_DIR%\composeTree.py %projDir% >> %logDir%\composeTree.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Composing tree succeeded
    goto PostProcessingTree
) else (
    echo Composing tree failed with exitCode %ERRORLEVEL%, terminating script
    goto end
)

:PostProcessingTree
echo STEP 4: Post processing tree
%PYTHON_EXE% %SCRIPT_DIR%\postProcessingTree.py %projDir% >> %logDir%\postProcessingTree.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Post processing tree succeeded
    goto splitClassification
) else (
    echo Post processing tree failed with exitCode %ERRORLEVEL%, terminating script
    goto end
)

:splitClassification
echo STEP 5: Split classification
%PYTHON_EXE% %SCRIPT_DIR%\splitClassification.py %projDir% >> %logDir%\splitClassification.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Split classification succeeded
    goto blenderLabel
) else (
    echo Split classification failed with exitCode %ERRORLEVEL%, terminating script
    goto end
)

:blenderLabel
echo STEP 6: Blender label
%BLENDER_EXE% -b -P %SCRIPT_DIR%\labelSkeleton.py >> %logDir%\labelSkeleton.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Blender label succeeded
    goto end
) else (
    echo Blender label failed with exitCode %ERRORLEVEL%, terminating script
    goto end
)

:end
set end=%time%
set options="tokens=1-4 delims=:.,"
for /f %options% %%a in ("%start%") do set start_h=%%a&set /a start_m=100%%b %% 100&set /a start_s=100%%c %% 100&set /a start_ms=100%%d %% 100
for /f %options% %%a in ("%end%") do set end_h=%%a&set /a end_m=100%%b %% 100&set /a end_s=100%%c %% 100&set /a end_ms=100%%d %% 100
set /a hours=%end_h%-%start_h%
set /a mins=%end_m%-%start_m%
set /a secs=%end_s%-%start_s%
set /a ms=%end_ms%-%start_ms%
if %ms% lss 0 set /a secs = %secs% - 1 & set /a ms = 100%ms%
if %secs% lss 0 set /a mins = %mins% - 1 & set /a secs = 60%secs%
if %mins% lss 0 set /a hours = %hours% - 1 & set /a mins = 60%mins%
if %hours% lss 0 set /a hours = 24%hours%
if 1%ms% lss 100 set ms=0%ms%
:: 计算时间并输出
set /a totalsecs = %hours%*3600 + %mins%*60 + %secs%
echo execution time %hours%:%mins%:%secs%.%ms% (%totalsecs%.%ms%s total)

ENDLOCAL