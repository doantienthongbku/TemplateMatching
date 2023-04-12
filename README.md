# Template Matching

## Setup environment

Create environment and install pytorch
```
python3 -m venv env
source env/bin/activate
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
```
Install requirements
```
pip install -r requirements.txt
```

## Running

```
python3 main.py [image_path] [template_path]
```

For example

```
python3 main.py sample/sample1.jpg sample/template1.png
```