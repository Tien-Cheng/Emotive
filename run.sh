eval "$(conda shell.bash hook)"
conda activate sp-dl-ca2
export FLASK_ENV=development
export DEVELOPMENT="config_dev.cfg"
python -m flask run