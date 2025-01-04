## rune-solver
Open the terminal(cmd) and run all of the following:

Manage the environment using pyenv
https://github.com/pyenv-win/pyenv-win
Follow the instructions and install version 3.10.0

Use these 2 commands to create a virtualenv called .venv
pyenv exec pip install virtualenv
pyenv exec python -m virtualenv .venv

For Windows:
.venv\Scripts\activate

You will notice the prefix change to (.venv) in the terminal

pip install -r requirements.txt

# To troubleshoot to make sure you have Torch working with your cuda drivers
in the (.venv) terminal session type in:
1. python
2. import torch
3. print(torch.__version__)            # Should have the letters "+cu" in it to indicate cuda torch version installed
4. print(torch.cuda.is_available())    # Should return True if CUDA is available
5. print(torch.cuda.current_device())  # Should show the device ID (typically 0)
6. print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Should show GPU name

pay attention to the version and make sure it has the words +cu in it with a number at the end


## Start the program with
python -m main

or

create a shortcut and put in(with the required modifications for the cd path):

C:\Windows\System32\cmd.exe /k "cd C:\Users\path\to\rune-solver & .venv\Scripts\activate & python main.py"