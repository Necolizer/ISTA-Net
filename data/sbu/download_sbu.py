import os
import requests
from tqdm import tqdm
import argparse

# Links
# See http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt
clean_version = [
    'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s02.zip',
    'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s03.zip',
    'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s07.zip',
    'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s01.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s03.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s06.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s07.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s02.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s04.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s05.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s06.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s02.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s03.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s06.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s02.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s03.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s02.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s03.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s04.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s01.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s03.zip',
]

noisy_version = [
    'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s02_noisy.zip',
    'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s03_noisy.zip',
    'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s07_noisy.zip',
    'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s01_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s03_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s06_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s07_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s02_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s04_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s05_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s06_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s02_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s03_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s06_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s02_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s03_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s02_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s03_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s04_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s01_noisy.zip',
	'http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s03_noisy.zip',
]

def download_files(links, save_dir):
    for link in links:
        file_name = link.split('/')[-1]  # Extract the file name from the URL
        save_path = os.path.join(save_dir, file_name)
        response = requests.get(link, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Get the file size from the response headers
            file_size = int(response.headers.get('Content-Length', 0))

            # Create a progress bar using tqdm
            progress_bar = tqdm(total=file_size, unit='B', unit_scale=True)

            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            progress_bar.close()
            print(f"Downloaded: {file_name}")
        else:
            print(f"Failed to download: {file_name}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download SBU-Kinect-Interaction dataset version 2.0")
    parser.add_argument('--version', type=str, help = 'clean/noisy', default='noisy')
    parser.add_argument('--savedir', type=str, help = 'Download Directory', default='./SBU-Kinect-Interaction/Noisy')

    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    links = clean_version if args.version.lower() == 'clean' else noisy_version

    download_files(links, args.savedir)
