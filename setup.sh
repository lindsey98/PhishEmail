conda create -n emailenv python=3.10
conda activate emailenv
sudo apt-get install poppler-utils  # For Linux (required by pdf2image)
sudo apt-get install wkhtmltopdf
pip install -r requirements.txt
gdown --id 10Gu1zlRiCS5ICglNJQGq-2wpPPLKqcZ8 -O checkpoints.zip
unzip checkpoints.zip -d checkpoints/


