import requests
import os
from tqdm import tqdm
ACCESS_TOKEN = os.environ.get('ZENODO_ACCESS')

headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}
nhdf_url = "https://zenodo.org/records/5549971/files/cell_lines_NHDF.zip"
u87_url = "https://zenodo.org/records/5549971/files/cell_lines_U87.zip"
url = "https://zenodo.org/records/4067741/files/foci_detection_testing_subset.zip"
out_file = "cell_lines_U87.zip"


response = requests.get(u87_url, headers=headers, stream=True)


# Sizes in bytes.
total_size = int(response.headers.get("content-length", 0))
block_size = 2048



if __name__ == "__main__":
    with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
        with open(out_file, "wb") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
    print(f"Downloaded {out_file}")
