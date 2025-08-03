from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import time
import html

# Setup headless Chrome
options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 15)

driver.get("https://www.cnbc.com/press-releases/")

#Load all articles by pressing "Load More"
previous_count = 0
tries = 0
max_tries = 5

while True:
    soup = BeautifulSoup(driver.page_source, "html.parser")
    current_links = soup.select("a.Card-title")
    current_count = len(current_links)

    if current_count > previous_count:
        previous_count = current_count
        tries = 0
    else:
        tries += 1
        if tries >= max_tries:
            print("Done expanding.")
            break
        time.sleep(2)
        continue

    try:
        load_more = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.LoadMoreButton-loadMore")))
        driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
        time.sleep(1)
        load_more.click()
        time.sleep(3)
    except:
        print("'Load More' not found or not clickable.")
        break


soup = BeautifulSoup(driver.page_source, "html.parser")
press_links = []
seen = set()

for a in soup.select("a.Card-title"):
    href = a.get("href")
    title = a.get_text(strip=True)

    if not href.endswith(".html"):
        continue
    if not href.startswith("http"):
        href = "https://www.cnbc.com" + href
    if href in seen:
        continue

    seen.add(href)
    press_links.append({"title": title, "url": href})

print(f"Found {len(press_links)} press releases.")


results = []

for i, pr in enumerate(press_links, 1):
    try:
        driver.get(pr["url"])
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(1.5)
        description = "No description"
        try:
            js_data = driver.execute_script("return window.__s_data;")

            layouts = js_data.get("page", {}).get("page", {}).get("layout", [])
            for layout in layouts:
                for column in layout.get("columns", []):
                    for module in column.get("modules", []):
                        if module.get("name") == "articleBody":
                            description = module["data"].get("articleBodyText", "No description")
                            raise StopIteration 
        except StopIteration:
            pass
        except Exception as e:
            description = f"Error extracting from JS: {e}"


        description = html.unescape(description)
        start_phrase = "Following is the unofficial transcript"
        start_index = description.find(start_phrase)

        if start_index != -1:
            description = description[start_index:]
        else:
            if "WHEN:" in description and "WHERE:" in description:
                parts = description.split("WHERE:", 1)
                description = parts[1].strip()

        date_text = "Not found"
        soup = BeautifulSoup(driver.page_source, "html.parser")
        time_tag = soup.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            try:
                dt_obj = datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00"))
                date_text = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except:
                date_text = time_tag.get_text(strip=True)

    except Exception as e:
        description = f"Error: {e}"
        date_text = "Error"

    print(f"{i}/{len(press_links)}: {description[:60]} — Date: {date_text}")
    results.append([
        pr["title"],
        pr["url"],
        description,
        date_text
    ])

# Save to Excel
df = pd.DataFrame(results, columns=["Title", "URL", "Description", "Date"])
output_file = "CNBC_All_Press_Releases_With_Clean_Descriptions.xlsx"
df.to_excel(output_file, index=False)
print(f"Data saved to: {output_file}")

driver.quit()
