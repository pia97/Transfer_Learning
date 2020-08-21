start_time=`date +%s`
python transfer_learning.py
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s. > /home/alamayreh/pia/time.log
