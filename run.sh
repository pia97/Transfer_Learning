start_time=`date +%s`
python transfer_learning.py
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s. > /home/pia/Dokumente/Studium/AI/Transfer Learning Server/time.log
