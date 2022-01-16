eval "$(conda shell.bash hook)"
conda activate sp-dl-ca2
export FLASK_ENV=development
python -m flask run