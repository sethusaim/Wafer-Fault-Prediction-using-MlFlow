sudo apt update && sudo apt-get update

wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh

bash Anaconda3-2021.11-Linux-x86_64.sh

export PATH=~/anaconda3/bin:$PATH

conda init bash

echo "Close the connection and reconnect"