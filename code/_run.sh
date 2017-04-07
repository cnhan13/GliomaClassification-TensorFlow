NUM_SET=9
IS_TUMOR_CROPPED=1
SET_CHAR="a"
TEN_THOUSANDS_STEPS=5
IS_RESET=0
for ((i=2; i<=$NUM_SET; i++))
do
  date
  echo "Training set $i, tumor_cropped $IS_TUMOR_CROPPED, set_char $SET_CHAR"
  train_out="p5c1_train_set$i"
  train_out+="_tumor$IS_TUMOR_CROPPED"
  train_out+="_$SET_CHAR"
  python p5c1_train.py $i $IS_TUMOR_CROPPED $IS_RESET $SET_CHAR | tee $train_out
  date
  echo "Evaluating set $i, tumor_cropped $IS_TUMOR_CROPPED"
  eval_out="p5c1_eval_set$i"
  eval_out+="_tumor$IS_TUMOR_CROPPED"
  eval_out+="_$SET_CHAR"
  python p5c1_eval.py $i $IS_TUMOR_CROPPED $IS_RESET $SET_CHAR | tee $eval_out
done
