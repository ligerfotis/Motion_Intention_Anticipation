# Motion Intention Anticipation

### Description

LSTM Model that predicts the class of anticipated human's motion.


### Python Environment set-up
Python3
Install venv for python3 virtual environments: 

    sudo apt install -y python3-venv

Go to project directory: 

    cd transf_chatbot

Create a python virtual environment: 

    python3 -m venv project_env

Activate python: 

    source project_env/bin/activate

Install requirements: 

    pip3 install -r requirements.txt


### Running the Code (Python 3.6)
        
    python model.py


###### Important: Remove Models after each training session, otherwise the code will initialize the new models with the old NN variables 