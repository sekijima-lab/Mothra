@echo off
setlocal

for /f "usebackq" %%X in (`docker-compose ps -q`) do set DCPS=%%X
if ""%DCPS%%""=="""" (goto __LAUNCH__) else (goto __SHUTDOWN__)

:__LAUNCH__
echo Launch...
docker-compose build --parallel
start http://localhost:8080
docker-compose up
goto __END__

:__SHUTDOWN__
echo Shutdown...
docker-compose down
goto __END__

:__END__