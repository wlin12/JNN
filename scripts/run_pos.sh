sh scripts/download_posdata.sh

java -Xmx2g -cp jnn.jar:libs/* jnn.functions.nlp.app.pos.PosTagger -lr 0.3 -batch_size 100 -validation_interval 5 -threads 1 -train_file twpos-data-v0.3/oct27.splits/oct27.train -validation_file twpos-data-v0.3/oct27.splits/oct27.dev -test_file twpos-data-v0.3/oct27.splits/oct27.test -input_format conll-0-1 -word_features characters -context_model window -iterations 100000 -output_dir /models/pos_model -sequence_activation 2 -word_dim 50 -char_dim 50 -char_state_dim 150  -context_state_dim 150 -update momentum -nd4j_resource_dir nd4j_resources/
