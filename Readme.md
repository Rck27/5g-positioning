pip install -r requirements.txt


#train the models


python train_master.py


#shows available index to generate

python create_json.py --list


#to create dummy static data at certain index

python create_json.py --index 42


#run the program, detecting for single location from json

python  main.py --input test_loc_42.json