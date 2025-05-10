# DocUNet Web Deployment

A Streamlit web application for document invoice rectification using the DocUNet algorithm.

## Features

- Upload and process distorted document images
- Real-time visualization of the rectification process
- Quality metrics (SSIM and PSNR) for result evaluation
- Download rectified documents

## Setup

1. Clone the repository:
```bash
git clone https://github.com/itrinia/docunet-webdeploy.git
cd docunet-webdeploy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
2. Upload a document image using the file uploader
3. Adjust rectification parameters if needed
4. Click "Rectify Document" to process the image
5. View and download the results

## Developer

Ileene Trinia Santoso  
Universitas Ciputra Surabaya

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
