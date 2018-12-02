input_number=$1

input_file=./input.$input_number
output_file=./output.$input_number

cat $input_file | ./my_set > $output_file && less $output_file
