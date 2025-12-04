Environment set up
python3 -m venv ~/venvs/genai
source ~/venvs/genai/bin/activate
pip install -r requirements.txt


data generation

python ./src/fine-tuning+eval.py --num_epochs 5 --max_seq_length 4096 --eval_every_epochs 1 --val_ratio 0.1 --test_ratio 0.1 --output_dir result3
