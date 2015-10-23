sh scripts/download_wikidata.sh

java -Xmx1g -cp jnn.jar:libs/* jnn.functions.nlp.app.lm.LSTMLanguageModel -batch_size 10 -iterations 1000 -lr 0.1 -output_dir ./models/lm_model -softmax_function word -test_file wiki/wiki.test.en -threads 1 -train_file wiki/wiki.train.en -validation_file wiki/wiki.dev.en -validation_interval 10 -word_dim 50 -char_dim 50 -char_state_dim 150 -lm_state_dim 150 -word_features characters -nd4j_resource_dir nd4j_resources -update momentum
