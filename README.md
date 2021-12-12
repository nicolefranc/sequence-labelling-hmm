# POS Tagging using Hidden Markov Model
## Part 1
```bash
# To run the model
python3 part1.py

# To evaluate
python3 EvalScript/evalResult.py ES/dev.out ES/1/dev.prediction 
python3 EvalScript/evalResult.py RU/dev.out RU/1/dev.prediction 
```

## Part 2
```bash
# To run the model
python3 part2.py

# To evaluate
python3 EvalScript/evalResult.py ES/dev.out ES/2/dev.prediction 
python3 EvalScript/evalResult.py RU/dev.out RU/2/dev.prediction
```

## Part 3
```bash
# To run the model
python3 part3.py

# To evaluate
python3 EvalScript/evalResult.py ES/dev.out ES/3/dev.prediction 
python3 EvalScript/evalResult.py RU/dev.out RU/3/dev.prediction
```

## Part 4
```bash
# To run the model
python3 part4.py

# To evaluate
# - using the dev set
python3 EvalScript/evalResult.py ES/dev.out ES/4/dev.p4.out 
python3 EvalScript/evalResult.py RU/dev.out RU/4/dev.p4.out

# - using the test set
python3 EvalScript/evalResult.py ES/dev.out ES/4/test.p4.out 
python3 EvalScript/evalResult.py RU/dev.out RU/4/test.p4.out
```