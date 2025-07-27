# EETICS-5G POSITIONING


### install requirement

```python
pip install -r requirements.txt
```

### train the models

```python
python train_master.py
```

### shows available index to generate

```python
python create_json.py --list
```

### create dummy static data at certain index

```python
python create_json.py --index 42
```

### JSON structure

see test_loc_X.json


### run the program, detecting for single location from json

```python
python  main.py --input test_loc_42.json
```
