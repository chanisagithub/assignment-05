import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# --- CONFIGURATION ---
BASE_URL = "https://riyasewana.com"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
TARGET_MAKES = ['toyota', 'nissan', 'honda',
                'suzuki', 'daihatsu', 'kia',
                'mitsubishi', 'ford', 'mazda',
                'hyundai', 'perodua']

# TARGET_MAKES = ['kia']

MAX_PAGES_PER_MAKE = 50  # Set to 50 or 100 for full scraping

def get_listing_urls(make):
    """Iterates through pagination for a specific make to get individual ad URLs."""
    listing_urls = []
    page = 1
    
    while page <= MAX_PAGES_PER_MAKE:
        # Search URL pattern: https://riyasewana.com/search/toyota/1
        url = f"{BASE_URL}/search/{make}/{page}"
        print(f"--> Scanning {make} (Page {page})...")
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                print(f"Failed to load page {page} (Status: {resp.status_code})")
                break
            
            soup = BeautifulSoup(resp.content, 'html.parser')
            
            # SELECTOR 1: Find all ad containers (li class="item")
            items = soup.find_all('li', class_='item')
            
            if not items:
                print(f"No items found on page {page}. Stopping {make}.")
                break
                
            for item in items:
                # SELECTOR 2: Find the link inside the h2 tag
                h2_tag = item.find('h2')
                if h2_tag:
                    a_tag = h2_tag.find('a')
                    if a_tag and 'href' in a_tag.attrs:
                        listing_urls.append(a_tag['href'])
            
            page += 1
            # Polite delay to avoid IP ban
            time.sleep(random.uniform(1.0, 2.5))
            
        except Exception as e:
            print(f"Error on {url}: {e}")
            break
            
    return list(set(listing_urls)) # Remove duplicates



def clean_currency(value):
    """Utility: Removes 'Rs.', 'Negotiable', commas and converts to integer to preserve all zeros."""
    if not value: return None
    # Remove non-numeric characters (keep only digits)
    clean_val = re.sub(r'\D', '', str(value))
    try:
        return int(clean_val) if clean_val else None
    except ValueError:
        return None

def clean_number(value):
    """Utility: Removes 'km', 'cc', commas and converts to int."""
    if not value: return None
    clean_val = re.sub(r'\D', '', str(value)) # Remove all non-digits
    try:
        return int(clean_val) if clean_val else None
    except ValueError:
        return None

def scrape_vehicle_details(url):
    """Extracts and CLEANS data from the detail page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # data = {'url': url}
        data = {}
        
        # 1. EXTRACT TITLE
        h1 = soup.find('h1')
        data['Title'] = h1.get_text(strip=True) if h1 else "N/A"
        
        # 2. EXTRACT PRICE (Strategy: Find the Price row in the table)
        price_text = None
        tables = soup.find_all('table', class_='moret')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                # Check if this row has Price label
                for i, col in enumerate(cols):
                    if 'Price' in col.get_text(strip=True) and i + 1 < len(cols):
                        # Get the next td which contains the price value
                        price_td = cols[i + 1]
                        price_span = price_td.find('span', class_='moreph')
                        if price_span:
                            price_text = price_span.get_text(strip=True)
                        else:
                            price_text = price_td.get_text(strip=True)
                        break
                if price_text:
                    break
            if price_text:
                break
        
        # Clean the price immediately
        data['Price'] = clean_currency(price_text)
        
        # 3. EXTRACT TABLE DATA (Strategy: Iterate ALL tables)
        # Riyasewana puts specs in a table. We check all rows of all tables.
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                
                # Process label-value pairs (handles both 2-column and 4-column rows)
                pairs = []
                if len(cols) == 2:
                    pairs = [(cols[0], cols[1])]
                elif len(cols) == 4:
                    # Two pairs: (col0, col1) and (col2, col3)
                    pairs = [(cols[0], cols[1]), (cols[2], cols[3])]
                
                for label_col, value_col in pairs:
                    key_text = label_col.get_text(strip=True).replace(':', '').strip().lower()
                    val_text = value_col.get_text(strip=True)
                    
                    # Map common keys to standard column names and CLEAN them
                    if 'make' in key_text and 'model' not in key_text:
                        data['Make_Detail'] = val_text
                    elif 'model' in key_text:
                        data['Model'] = val_text
                    elif 'yom' in key_text or ('year' in key_text and 'manufacture' not in key_text):
                        data['YOM'] = clean_number(val_text)
                    elif 'mileage' in key_text:
                        data['Mileage_km'] = clean_number(val_text)
                    elif 'engine' in key_text and 'cc' in key_text.lower() + val_text.lower():
                        data['Engine_cc'] = clean_number(val_text)
                    elif 'fuel' in key_text:
                        data['Fuel_Type'] = val_text.strip()
                    elif 'gear' in key_text or 'transmission' in key_text:
                        data['Transmission'] = val_text.strip()
                    elif 'option' in key_text or 'feature' in key_text:
                        data['Options'] = val_text
                    elif 'contact' in key_text or 'phone' in key_text:
                        data['Contact'] = val_text
                    elif 'city' in key_text or 'location' in key_text:
                        data['Location'] = val_text
                    elif 'condition' in key_text:
                        data['Condition'] = val_text
                    elif 'body' in key_text:
                        data['Body_Type'] = val_text
                    elif 'color' in key_text or 'colour' in key_text:
                        data['Color'] = val_text
                    
        return data

    except Exception as e:
        print(f"Skipping {url} due to error: {e}")
        return None

# --- EXECUTION ---
full_dataset = []

print("Starting Scrape...")
for make in TARGET_MAKES:
    urls = get_listing_urls(make)
    print(f"Found {len(urls)} vehicles for {make}. Extracting details...")
    
    for i, link in enumerate(urls):
        vehicle_data = scrape_vehicle_details(link)
        if vehicle_data:
            vehicle_data['Make'] = make # Add the make explicitly
            full_dataset.append(vehicle_data)
        
        # Print progress every 10 items
        if i % 10 == 0:
            print(f"Scraped {i}/{len(urls)}...")
            
        time.sleep(0.5) # Fast but safe delay

# Save to CSV
df = pd.DataFrame(full_dataset)
df.to_csv('riyasewana_raw_data.csv', index=False)
print(f"Done! Saved {len(df)} rows to riyasewana_raw_data.csv")