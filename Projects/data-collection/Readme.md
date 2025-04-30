# วิธีการสร้าง Dataset

## ขั้นตอนการเตรียมข้อมูล

1. เปิด terminal แล้ว install libraries ด้วย
   ```bash
   pip install -r requirements.txt
   ```

2. เปิดไฟล์ `dataset/weather/weather_points.csv` เพื่อดู coordinate ที่ต้องดึงข้อมูลมาทั้งหมด

3. เปิด [Open-meteo-historical](https://open-meteo.com/en/docs/historical-weather-api?daily=weather_code,temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant,cloud_cover_mean,cloud_cover_max,cloud_cover_min,relative_humidity_2m_mean,relative_humidity_2m_min,relative_humidity_2m_max,surface_pressure_mean,surface_pressure_max,surface_pressure_min,visibility_mean,visibility_min,visibility_max,winddirection_10m_dominant,wind_speed_10m_mean,wind_speed_10m_min,vapour_pressure_deficit_max&timezone=Asia%2FBangkok&start_date=2023-01-01&end_date=2024-12-31) เพื่อเข้าไปดึงข้อมูลสภาพอากาศย้อนหลัง
   และ [Open-meteo-historical-forecast](https://open-meteo.com/en/docs/historical-forecast-api?daily=weather_code,temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_direction_10m_dominant,cloud_cover_mean,cloud_cover_max,cloud_cover_min,relative_humidity_2m_mean,relative_humidity_2m_min,relative_humidity_2m_max,surface_pressure_mean,surface_pressure_max,surface_pressure_min,visibility_mean,visibility_min,visibility_max,winddirection_10m_dominant,wind_speed_10m_mean,wind_speed_10m_min,vapour_pressure_deficit_max&timezone=Asia/Bangkok&start_date=2023-01-01&end_date=2024-12-31) เพื่อเข้าไปดึงข้อมูล**ทำนาย**สภาพอากาศย้อนหลัง

4. เอา lat, lng ที่อยู่ในขั้นตอนที่ 2 ไปกรอกใน web ทั้ง 2 แล้ว load CSV มา

5. เก็บ ข้อมูลสภาพอากาศย้อนหลัง CSV ที่ได้ไว้ใน `dataset/weather/<station_id>` แล้วแก้ชื่อไฟล์โดยเพิ่มทิศต่อท้าย (ดูตัวอย่างใน `dataset/weather/39T`)
   
6. เก็บ ข้อมูล**ทำนาย**สภาพอากาศย้อนหลัง CSV ที่ได้ไว้ใน `dataset/Weather_forecast/<station_id>` แล้วแก้ชื่อไฟล์โดยเพิ่มทิศต่อท้าย (ดูตัวอย่างใน `dataset/Weather_forecast/39T`)

## การสร้าง Dataset

6. เปิดไฟล์ `create_dataset.ipynb`
   
7. แก้ directory CSV ที่ download มาใน code นี้:
   ```python
   pm25_files = sorted(glob.glob("dataset/PM2.5/PM2.5(*.xlsx")) 
   fire_file = "dataset/Fire/fire_archive_M-C61_606028.csv"
   weather_files = sorted(glob.glob("dataset/weather/39T/*.csv")) 

8. แก้ไขค่าต่อไปนี้ให้เป็น station_id ที่เรากำลังจะสร้าง และ lat, lng ของ center ของ station นั้นๆ:
   ```python
   station_id = '39T'
   station_lat = 18.427180  # center
   station_lng = 99.757746  # center
   start_date = '2019-01-01'
   end_date = '2024-12-31'
   ```

9.  รัน code จนครบ จะได้ dataset อยู่ใน `dataset/Full` เพื่อนำไปใช้ต่อไป

10. ทำแบบนี้สำหรับทั้ง `weather` และ `Weather_forecast` 
    ปล. สำหรับ `Weather_forecast` อย่าลืมมาแก้ output file ให้มี forecast ตามหลัง ID แบบนี้ `save_combined_dataset(combined_df, output_file=f'dataset/Full/combined_pm25_{station_id}_forecast_dataset.csv')`