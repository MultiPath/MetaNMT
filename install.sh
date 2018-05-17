wget http://sqlite.org/2013/sqlite-autoconf-3080100.tar.gz
tar xvfz sqlite-autoconf-3080100.tar.gz --no-same-owner
cd sqlite-autoconf-3080100
./configure
make
make install

curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
export PATH="/root/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo 'export PATH="/root/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

pyenv install 3.6.5
pyenv local 3.6.5

pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip install tensorflow
pip install tqdm
pip install tensorboardX
pip install nltk

cd ./tools/torchtext/
python setup.py install
cd ../rev/
python setup.py install
git config --global user.email "thomagram@gmail.com"
git config --global user.name "Jiatao Gu"
