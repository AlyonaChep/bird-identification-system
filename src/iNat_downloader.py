import requests
import os
import urllib.request

# Назва виду (можна також ID таксону)
species_name = "Cyanistes caeruleus caeruleus"
output_dir = "blue_tit"
num_images = 200

# Passer domesticus - European House Sparrow
# Cyanistes caeruleus caeruleus - Common Blue Tit
# Turdus merula merula - Western European Blackbird
# Parus major major - Common Great Tit
# Erithacus rubecula - European Robin

os.makedirs(output_dir, exist_ok=True)

def get_observations(species, per_page=30, page=1):
    url = f"https://api.inaturalist.org/v1/observations"
    params = {
        "q": species,
        "per_page": per_page,
        "page": page,
        "photos": "true",
        "quality_grade": "research",
        "order_by": "created_at",
    }
    return requests.get(url, params=params).json()

downloaded = 0
page = 1

while downloaded < num_images:
    data = get_observations(species_name, per_page=30, page=page)
    results = data.get("results", [])
    if not results:
        break

    for obs in results:
        if downloaded >= num_images:
            break
        photos = obs.get("photos", [])
        for photo in photos:
            if downloaded >= num_images:
                break
            url = photo["url"].replace("square", "original")  # максимальна якість
            filename = os.path.join(output_dir, f"{downloaded}.jpg")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded {filename}")
                downloaded += 1
            except:
                print(f"Failed to download: {url}")
    page += 1