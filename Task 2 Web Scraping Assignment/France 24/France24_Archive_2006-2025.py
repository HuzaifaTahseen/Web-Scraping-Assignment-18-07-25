from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def extract_description(driver, url):
    try:
        driver.get(url)
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        chapo = soup.select_one("p.t-content__chapo")
        if chapo:
            return chapo.get_text(strip=True)
        return ""
    except Exception as e:
        print(f"Failed to get description from {url}: {e}")
        return ""

def scrape_year(year):
    results = []
    base_year_url = f"https://www.france24.com/en/archives/{year}/"
    print(f"\nYear: {year}")

    driver = create_driver()
    try:
        driver.get(base_year_url)
        time.sleep(3)

        # Scroll to load more day links
        for _ in range(8):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        day_links = soup.select("a.o-archive-month__days__day__link")

        for link in day_links:
            day_url = link.get("href")
            day_date = link.text.strip()
            if not day_url.startswith("http"):
                day_url = "https://www.france24.com" + day_url
            print(f"{day_date}")

            try:
                driver.get(day_url)
                time.sleep(1.5)
                day_soup = BeautifulSoup(driver.page_source, "html.parser")
                articles = day_soup.select("ul.o-archive-day__list li.o-archive-day__list__entry a.a-archive-link")

                for a in articles:
                    title = a.get_text(strip=True)
                    article_url = a.get("href")
                    if not article_url.startswith("http"):
                        article_url = "https://www.france24.com" + article_url

                    description = extract_description(driver, article_url)

                    results.append({
                        "Title": title,
                        "Url": article_url,
                        "Date": day_date,
                        "Description": description
                    })

            except Exception as e:
                continue

    finally:
        driver.quit()

    return results

# Main loop
all_results = []
for y in range(2006, 2026):
    year_data = scrape_year(y)
    all_results.extend(year_data)
    pd.DataFrame(year_data).to_csv(f"France24_Archive_{y}.csv", index=False)

df = pd.DataFrame(all_results)
df.to_csv("france24_full_archive_2006_2025.csv", index=False)
print(f"\nTotal articles saved: {len(df)} → france24_full_archive_2006_2025.csv")
