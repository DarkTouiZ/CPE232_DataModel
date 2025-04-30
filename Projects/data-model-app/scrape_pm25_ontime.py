from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from datetime import datetime

def setup_driver(headless=True):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    return driver, WebDriverWait(driver, 20)

def select_region(driver, wait, region_name="ภาคเหนือ"):
    region_select = wait.until(EC.presence_of_element_located((By.XPATH, '//select[contains(@class, "custom-select")]')))
    region_select.click()
    for option in region_select.find_elements(By.TAG_NAME, 'option'):
        if region_name in option.text:
            option.click()
            print(f"Selected region: {region_name}")
            break
    time.sleep(0.5)

def select_stations(driver, wait, station_names):
    station_box = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'multiselect__tags')))
    station_box.click()
    time.sleep(0.5)
    
    station_options = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'multiselect__option')))
    selected_count = 0
    
    for name in station_names:
        for option in station_options:
            if name.strip() in option.text.strip():
                option.click()
                selected_count += 1
                time.sleep(0.2)
                break
    
    print(f"Selected {selected_count} stations")

def select_date(driver, date_id, target_date):
    date_input = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, date_id)))
    date_input.click()
    time.sleep(0.5)
    
    target_month_year = target_date.strftime("%B %Y").replace("January", "มกราคม").replace("February", "กุมภาพันธ์").replace("March", "มีนาคม").replace("April", "เมษายน").replace("May", "พฤษภาคม").replace("June", "มิถุนายน").replace("July", "กรกฎาคม").replace("August", "สิงหาคม").replace("September", "กันยายน").replace("October", "ตุลาคม").replace("November", "พฤศจิกายน").replace("December", "ธันวาคม")
    
    while True:
        month_header = driver.find_element(By.CLASS_NAME, "vdatetime-calendar__current--month").text
        if target_month_year in month_header:
            break
        nav_button = driver.find_element(By.CLASS_NAME, "vdatetime-calendar__navigation--previous")
        nav_button.click()
        time.sleep(0.3)
    
    target_day = str(target_date.day)
    days = driver.find_elements(By.CLASS_NAME, "vdatetime-calendar__month__day")
    for day in days:
        if day.text.strip() == target_day:
            day.click()
            break
    
    ok_button = driver.find_element(By.CLASS_NAME, "vdatetime-popup__actions__button--confirm")
    ok_button.click()
    print(f"Selected date: {target_date.strftime('%d-%m-%Y')}")

def set_date_range(driver, start_date, end_date):
    select_date(driver, "startDate", start_date)
    time.sleep(0.5)
    select_date(driver, "endDate", end_date)
    time.sleep(0.5)

def select_pollutant(driver, pollutant="PM2.5"):
    label = driver.find_element(By.CSS_SELECTOR, 'label[for="__BVID__31"]')
    label.click()
    print(f"Selected pollutant: {pollutant}")

def search_and_collect_data(driver, wait):
    driver.find_element(By.XPATH, '//button[contains(text(), "ตรวจสอบ")]').click()
    wait.until(EC.presence_of_element_located((By.XPATH, '//table')))
    time.sleep(1)
    
    all_data = []
    
    header_elements = driver.find_elements(By.CSS_SELECTOR, 'thead th')
    headers = [header.text for header in header_elements]
    all_data.append(headers)
    
    page_num = 1
    total_rows = 0
    
    while True:
        rows = driver.find_elements(By.CSS_SELECTOR, 'tbody tr')
        current_page_rows = len(rows)
        total_rows += current_page_rows
        
        print(f"Page {page_num}: Found {current_page_rows} rows")
        
        if not rows:
            break
            
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            row_data = []
            for col in cols:
                try:
                    i_tag = col.find_element(By.TAG_NAME, 'i')
                    row_data.append(i_tag.text)
                except:
                    row_data.append(col.text)
            
            if row_data and row_data[0] not in ['ค่าสูงสุด', 'ค่าน้อยสุด', 'ค่าเฉลี่ย', 'จำนวนที่เก็บค่าได้', 'จำนวนชั่วโมงทั้งหมด', 'เปอร์เซ็นต์การเก็บค่าได้']:
                all_data.append(row_data)
        
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, 'button[aria-label="Go to next page"]')
            if "disabled" in next_button.get_attribute("class") or not next_button.is_enabled():
                print("Reached last page")
                break
                
            driver.execute_script("arguments[0].click();", next_button)
            page_num += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"Error navigating: {e}")
            break
    
    print(f"Total data collected: {len(all_data)-1} rows")
    return all_data

def save_to_csv(data, filename):
    if len(data) > 1:
        df = pd.DataFrame(data)
        df = df.iloc[:, :9]
        df.columns = df.iloc[0]
        df = df[1:]
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"✅ Success: Saved data to {filename}")
        return True
    else:
        print("⚠️ No data found for the selected period")
        return False

def scrape_air_quality(start_date, end_date, region="ภาคเหนือ", stations=None, output_file="air_quality_data.csv", headless=True):
    if stations is None:
        stations = [
            "สำนักงานสาธารณสุขแม่สาย ต.เวียงพางคำ อ.แม่สาย, เชียงราย",
            "โครงการชลประทานนครสวรรค์ ต.ปากน้ำโพ อ.เมือง, นครสวรรค์",
            "โรงพยาบาลเฉลิมพระเกียรติ ต.ห้วยโก๋น อ.เฉลิมพระเกียรติ, น่าน",
            "สำนักงานทรัพยากรธรรมชาติและสิ่งแวดล้อมจังหวัดแม่ฮ่องสอน ต.จองคำ อ.เมือง, แม่ฮ่องสอน",
            "โรงพยาบาลส่งเสริมสุขภาพตำบลท่าสี ต.บ้านดง อ.แม่เมาะ, ลำปาง",
            "ศูนย์การศึกษานอกโรงเรียน ต.แม่ปะ อ.แม่สอด, ตาก"
        ]
    
    driver, wait = setup_driver(headless)
    try:
        print("Starting air quality data scraping...")
        driver.get("http://air4thai.pcd.go.th/webV3/#/History")
        time.sleep(1)
        
        select_region(driver, wait, region)
        select_stations(driver, wait, stations)
        set_date_range(driver, start_date, end_date)
        select_pollutant(driver)
        data = search_and_collect_data(driver, wait)
        success = save_to_csv(data, output_file)
        
        return success
    finally:
        driver.quit()

if __name__ == "__main__":
    start_date = datetime(2025, 3, 31)
    end_date = datetime(2025, 4, 30)
    scrape_air_quality(start_date, end_date, output_file="air_quality_northern.csv")