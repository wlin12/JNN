if [ ! -e twpos-data-v0.3.tgz ]; then
  wget https://ark-tweet-nlp.googlecode.com/files/twpos-data-v0.3.tgz
  tar -zxvf twpos-data-v0.3.tgz
else
  echo "file twpos-data-v0.3.tgz already exists"
fi


