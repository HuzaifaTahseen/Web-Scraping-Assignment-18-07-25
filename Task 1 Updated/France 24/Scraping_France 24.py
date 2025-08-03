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
import time

try:
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

    articles_data = []
    base_url = "https://www.france24.com"

    for info_div in soup.find_all("div", class_="article__infos"):
        a_tag = info_div.find("a", href=True)
        if not a_tag:
            continue

        article_url = a_tag["href"]
        if not article_url.startswith("http"):
            article_url = base_url + article_url

        h2_tag = a_tag.find("h2")
        title = h2_tag.get_text(strip=True) if h2_tag else a_tag.get_text(strip=True)

        date_span = info_div.find("span", class_="date")
        date = date_span.get_text(strip=True) if date_span else datetime.today().strftime("%Y-%m-%d")

        try:
            driver.get(article_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "t-content__body"))
            )
            article_soup = BeautifulSoup(driver.page_source, "html.parser")
            paragraphs = article_soup.select("div.t-content__body p")
            full_description = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

            if not full_description:
                full_description = "No Description Found"
        except Exception as e:
            full_description = f"Failed to load article: {e}"

        articles_data.append([title, article_url, full_description, date])
        print(f"Scraped: {title}")

    driver.quit()

    # Save to Excel
    df = pd.DataFrame(articles_data, columns=["Title", "URL", "Description", "Date"])
    excel_file = f"Data Scraped From France 24.xlsx"

    try:
        df.to_excel(excel_file, index=False)
        print(f"\nData has been successfully exported to: {excel_file}")
    except Exception as e:
        print("\nFailed to save Excel file.")
        print(f"Error: {e}")

except Exception as e:
    print("\nScript failed due to unexpected error.")
    print(f"Error: {e}")
