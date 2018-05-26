python ../PyModels/prepare_relevancy_dataset.py
if errorlevel 1 goto Error

python ../PyModels/prepare_wordchar_dataset.py
if errorlevel 1 goto Error

python ../PyModels/prepare_qa_dataset.py
if errorlevel 1 goto Error


echo Well done
goto:eof

:Error
echo Error
goto:eof

