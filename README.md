#### Repositorio para guardar codigo perteneciente al proyecto "Detección de cianobacterias mediante sensado remoto e inteligencia artificial"

## Requirements
<ul>
  <li>Ubuntu 18.04</li>
  <li>Anaconda</li>
</ul>

## Setup

### Environment
`conda env create -f linux_conda_env.yml`

### snappy
It is necessary to install the SNAP python API (snappy). Start by changing to the home directory of your current user. In my case: `cd /home/ubuntu`

`wget https://download.esa.int/step/snap/8.0/installers/esa-snap_all_unix_8_0.sh ; sh esa-snap_all_unix_8_0.sh`

Go through the installation and choose to configure SNAP for use with python but then choose not to run SNAP Desktop and configure Python.

`cd /home/ubuntu/snap/bin ; ./snappy-conf /home/ubuntu/anaconda3/envs/cyano_venv/bin/python`

`cd /home/ubuntu/.snap/snap-python/snappy/`

`/home/ubuntu/anaconda3/envs/cyano_venv/bin/python setup.py install`

### Use conda env in jupyter notebooks
`pip install ipykernel ; python -m ipykernel install --user --name=cyano_venv`

### OpenCV dependencies
`apt-get install ffmpeg libsm6 libxext6  -y`