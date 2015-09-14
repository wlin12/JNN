if [ ! -e wiki.gz ]; then
  wget https://www.l2f.inesc-id.pt/~wlin/wiki.gz
  tar -zxvf wiki.gz
else
  echo "file wiki.gz already exists"
fi


