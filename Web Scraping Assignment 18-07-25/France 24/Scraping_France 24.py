from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd


# Setup headless Chrome
options = Options()
options.add_argument("--headless=new") 
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)


url = "https://www.france24.com/en/"
driver.get(url)


WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CLASS_NAME, "article__infos"))
)


soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()


articles_data = []

for info_div in soup.find_all("div", class_="article__infos"):

    a_tag = info_div.find("a", href=True)
    if not a_tag:
        continue

    url = a_tag["href"]
    if not url.startswith("http"):
        url = "https://www.france24.com" + url

    h2_tag = a_tag.find("h2")
    title = h2_tag.get_text(strip=True) if h2_tag else a_tag.get_text(strip=True)

    desc_div = info_div.find("div", class_="article__chapo")
    description = desc_div.get_text(strip=True) if desc_div else "No Description"

    date_span = info_div.find("span", class_="date")
    date = date_span.get_text(strip=True) if date_span else datetime.today().strftime("%Y-%m-%d")

    articles_data.append([title, url, description, date])


# Convert list of articles to DataFrame
df = pd.DataFrame(articles_data, columns=["Title", "URL", "Description", "Date"])

# Export Data into Excel
excel_file = f"Data Scraped From France 24.xlsx"
df.to_excel(excel_file, index=False)

print(f"Data has been exported in ({excel_file})")