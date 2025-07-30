from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import time

# Setup headless Chrome
options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)


url = "https://www.cnbc.com/world/"
driver.get(url)
time.sleep(5)  


soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()


articles_data = []
seen_links = set()

for a_tag in soup.find_all("a", href=True):
    href = a_tag["href"]


    if not href.endswith(".html"):
        continue

    if not href.startswith("http"):
        href = "https://www.cnbc.com" + href

    if href in seen_links:
        continue
    seen_links.add(href)

    title = a_tag.get("title") or a_tag.get_text(strip=True)
    if not title or "watch now" in title.lower():
        continue

    # There is no description in the article links
    description = "No description"
    date = datetime.today().strftime("%Y-%m-%d")

    # Convert list of articles to DataFrame
    articles_data.append([title, href, description, date])

# Export Scrapped data to Excel
df = pd.DataFrame(articles_data, columns=["Title", "URL", "Description", "Date"])
excel_file = "Data Scraped From CNBC.xlsx"
df.to_excel(excel_file, index=False)

print(f"Data has been exported in ({excel_file})")
