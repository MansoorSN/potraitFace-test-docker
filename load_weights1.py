import os
import gdown

def download_weights():
    # Create directory if it does not exist
    weights_dir = "/root/.deepface/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    url = 'your_weights_url'
    output = os.path.join(weights_dir, 'your_weights_file')
    gdown.download(url, output, quiet=False)

if __name__ == "__main__":
    download_weights()
