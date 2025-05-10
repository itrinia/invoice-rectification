import os
import gdown
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile

def verify_h5_file(file_path):
    """Verify if a file is a valid HDF5 file that can be loaded by TensorFlow"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if this is a valid Keras model file
            if 'model_weights' in f or 'layer_names' in f:
                return True
            else:
                print("File appears to be H5 but not a Keras model")
                return False
    except Exception as e:
        print(f"Not a valid H5 file: {str(e)}")
        return False

def download_model():
    """Download the model file from Google Drive"""
    # Google Drive File ID
    GOOGLE_DRIVE_FILE_ID = "1aYNIwYh2R178-AYIXd1wo_ISa7jhFhd-"
    
    # Create a temporary file to store the model
    temp_model_path = os.path.join(tempfile.gettempdir(), 'docunet_model.h5')
    
    if os.path.exists(temp_model_path) and verify_h5_file(temp_model_path):
        print(f"Using cached model from {temp_model_path}")
        return temp_model_path
    
    print("Downloading model from Google Drive...")
    
    try:
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
        print(f"Downloading from: {url}")
        
        # Create a temporary directory first
        download_dir = os.path.join(tempfile.gettempdir(), 'docunet_downloads')
        os.makedirs(download_dir, exist_ok=True)
        temp_download_path = os.path.join(download_dir, 'docunet_model.h5')
        
        gdown.download(url, temp_download_path, quiet=False)
        
        if os.path.exists(temp_download_path):
            print(f"Download completed to {temp_download_path}")
            
            # Verify the downloaded file
            if verify_h5_file(temp_download_path):
                print("Downloaded file verified as valid H5 model")
                # Copy to the final location
                import shutil
                shutil.copy(temp_download_path, temp_model_path)
                print(f"Model copied to {temp_model_path}")
                return temp_model_path
            else:
                print("Downloaded file is not a valid model")
                return None
        else:
            print("Download failed - file not found after download")
            return None
            
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return None

def main():
    print("Starting model setup...")
    model_path = download_model()
    
    if model_path is None:
        print("Failed to download or verify the model. Please check your internet connection and try again.")
        return
    
    print("Model setup completed successfully!")
    print(f"Model is available at: {model_path}")

if __name__ == "__main__":
    main() 