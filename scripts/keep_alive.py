import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def keep_alive():
    app_url = os.environ.get("STREAMLIT_APP_URL")
    if not app_url:
        print("Error: STREAMLIT_APP_URL environment variable not set.")
        return

    print(f"Starting keep-alive check for: {app_url}")

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(app_url)
        print("Page loaded.")
        
        # Wait for potential content to load
        time.sleep(5)

        # Check for the "Yes, get this app back up!" button
        # Streamlit's "sleeping" page usually has a button with this text.
        # We'll try to find it by text content.
        try:
            xpath = "//button[contains(text(), 'Yes, get this app back up!')]"
            wake_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            print("Wake up button found!")
            wake_button.click()
            print("Clicked 'Yes, get this app back up!' button.")
            
            # Wait a bit to ensure the click registers
            time.sleep(5)
        except Exception as e:
            print("App appears to be already awake (or button not found).")
            # print(f"Debug info: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()
        print("Browser closed.")

if __name__ == "__main__":
    keep_alive()
