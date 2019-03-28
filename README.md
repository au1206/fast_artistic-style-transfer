# Fast Artistic File Transfer

This is the code for the paper

**[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/)**
<br>
[Justin Johnson](http://cs.stanford.edu/people/jcjohns/),
[Alexandre Alahi](http://web.stanford.edu/~alahi/),
[Li Fei-Fei](http://vision.stanford.edu/feifeili/)
<br>
Presented at [ECCV 2016](http://www.eccv2016.org/)



## Install

refreshing the repositories
```sh
sudo apt update
```

its wise to keep the system up to date! you can skip the following line if you not want to update all your software
```sh
sudo apt upgrade
```

installing python 3.6.5 and pip for it
```sh
sudo apt-get install python3-pip
```
installing python virtualenv
```sh
sudo pip install virtualenv
```

install virtualenv and application dependencies
```sh
virtualenv ~/fast_style_transfer/
source ~/fast_style_transfer/bin/activate
pip install -r requirements_python3.txt
```


## Usage

```sh
source ~/fast_style_transfer/bin/activate

``` 	

Once the virtualenv is set up, go throught train.py and change the paths of the dataset and folders and run to train.
