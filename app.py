import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. โหลดโมเดลที่บันทึกไว้ ---
# (ไฟล์ 'real_estate_model_v4.joblib' ต้องอยู่ในโฟลเดอร์เดียวกับ app.py)
try:
    model = joblib.load('real_estate_model_v4.joblib')
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดล 'real_estate_model_v4.joblib'!")
    st.write("กรุณารันโค้ด 'joblib.dump(model_rf, ...)' ในไฟล์ .ipynb ของคุณก่อน")
    st.stop()

# --- 2. กำหนดชื่อคอลัมน์ที่โมเดล V4 ต้องการ (สำคัญมาก!) ---
# (นี่คือลิสต์ที่ "ตรงกับ" RealEstate1.ipynb เซลล์ที่ 12)
NUMERIC_FEATURES = [
    'latitude', 'longitude', 'year_built', 'proj_area', 'nbr_floors', 'units',
    'Elevator', 'Parking', 'Security', 'CCTV', 'Pool', 'Sauna', 'Gym', 'Garden', 
    'Playground', 'Shop', 'Restaurant', 'Wifi',
    'dist_nearest_station', 'policy_rate', 'unemployment_count_k'
]
CATEGORICAL_FEATURES = ['district']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# (เราต้องรู้ว่าโมเดล V4 รู้จักเขตอะไรบ้าง)
# (สกัดมาจาก Train Set)
KNOWN_DISTRICTS = [
    'Bang Kapi', 'Huai Khwang', 'Bangkok Noi', 'Prawet', 'Bang Sue', 
    'Khlong Toei', 'Chatuchak', 'Lat Phrao', 'Bang Phlat', 'Phaya Thai', 
    'Phra Khanong', 'Sathon', 'Watthana', 'Pathum Wan', 'Suan Luang', 
    'Ratchathewi', 'Din Daeng', 'Bang Khen', 'Don Mueang', 'Thon Buri', 
    'Khlong San', 'Bang Rak', 'Yan Nawa', 'Chom Thong', 'Dusit', 
    'Phasi Charoen', 'Saphan Sung', 'Bang Kho Laem', 'Lak Si'
]

# --- 3. สร้างหน้าเว็บ (Web Interface) ---
st.set_page_config(page_title="ประเมินราคาอสังหาฯ ", layout="wide")
st.title("🏙️ แบบจำลองประเมินราคาอสังหาริมทรัพย์ ")
st.write("ป้อนปัจจัยต่างๆ เพื่อประเมิน 'value' (ราคา) โดยอ้างอิงจากโมเดล")

# สร้าง 2 คอลัมน์หลัก
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 ปัจจัยทำเลและกายภาพ")
    
    # Input ที่สำคัญ (จาก Top 5 ของ V4)
    in_nbr_floors = st.slider("จำนวนชั้น", 1, 60, 30, help="จำนวนชั้นทั้งหมดของอาคาร")
    in_year_built = st.slider("ปีที่สร้าง ", 1990, 2025, 2018, help="ปี ค.ศ. ที่สร้างเสร็จ")
    in_dist = st.number_input("ระยะทางไปรถไฟฟ้า (กม.)", 0.0, 20.0, 1.5, step=0.1)
    in_units = st.number_input("จำนวนยูนิต (units)", 1, 2000, 150)
    
    # Input ที่สำคัญ (เลือกเขต)
    in_district = st.selectbox("เลือกเขต", sorted(KNOWN_DISTRICTS))

    st.subheader("📈 ปัจจัยเศรษฐกิจ (ปัจจุบัน)")
    
    # Input ปัจจัยเศรษฐกิจ (สมมติว่าเราป้อนค่าปัจจุบัน)
    in_policy_rate = st.number_input("อัตราดอกเบี้ยนโยบาย (%)", 0.5, 5.0, 1.75, step=0.25, help="อ้างอิงจากไฟล์ FM_RT")

# === (นี่คือส่วนที่คุณต้องการ) สร้าง Checkbox ครบทั้ง 12 รายการ ===
st.subheader("🏊 ปัจจัยสิ่งอำนวยความสะดวก (Facilities)")
st.write("เลือกสิ่งอำนวยความสะดวกที่มีในโครงการ (ตรงตามโมเดล V4):")

# สร้าง 3 คอลัมน์ย่อย เพื่อให้ UI สวยงาม
f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    in_elevator = st.checkbox("ลิฟต์ (Elevator)", value=True)
    in_parking = st.checkbox("ที่จอดรถ (Parking)", value=True)
    in_security = st.checkbox("รปภ. (Security)", value=True)
    in_cctv = st.checkbox("CCTV", value=True)

with f_col2:
    in_pool = st.checkbox("สระว่ายน้ำ (Pool)", value=True)
    in_gym = st.checkbox("ฟิตเนส (Gym)", value=True)
    in_sauna = st.checkbox("ซาวน่า (Sauna)", value=False)
    in_garden = st.checkbox("สวน (Garden)", value=True)

with f_col3:
    in_playground = st.checkbox("สนามเด็กเล่น (Playground)", value=False)
    in_shop = st.checkbox("ร้านค้า (Shop)", value=False)
    in_restaurant = st.checkbox("ร้านอาหาร (Restaurant)", value=False)
    in_wifi = st.checkbox("Wifi ส่วนกลาง", value=True)


st.divider() # เส้นคั่น

# --- 4. สร้าง DataFrame สำหรับทำนาย (สำคัญที่สุด) ---
# เราต้องสร้าง DataFrame ที่มี 1 แถว และ "ทุกคอลัมน์" ที่โมเดลคาดหวัง

# สร้าง Dict ว่างๆ ด้วยค่า Default
input_data = {}
for col in NUMERIC_FEATURES:
    input_data[col] = 0.0 # ใส่ค่า 0 ไว้ก่อน
for col in CATEGORICAL_FEATURES:
    input_data[col] = "None" # ใส่ค่าว่างไว้ก่อน

# อัปเดต Dict ด้วยค่าจาก User Input
input_data.update({
    # ปัจจัยจาก col1
    'nbr_floors': in_nbr_floors,
    'year_built': in_year_built,
    'dist_nearest_station': in_dist,
    'units': in_units,
    'district': in_district,
    
    # ปัจจัยจาก col2
    'unemployment_count_k': in_unemployment,
    'policy_rate': in_policy_rate,
    
    # === (ส่วนที่แก้ไข) อัปเดต Facility ทั้ง 12 อย่างจาก User ===
    'Elevator': 1.0 if in_elevator else 0.0,
    'Parking': 1.0 if in_parking else 0.0,
    'Security': 1.0 if in_security else 0.0,
    'CCTV': 1.0 if in_cctv else 0.0,
    'Pool': 1.0 if in_pool else 0.0,
    'Sauna': 1.0 if in_sauna else 0.0,
    'Gym': 1.0 if in_gym else 0.0,
    'Garden': 1.0 if in_garden else 0.0,
    'Playground': 1.0 if in_playground else 0.0,
    'Shop': 1.0 if in_shop else 0.0,
    'Restaurant': 1.0 if in_restaurant else 0.0,
    'Wifi': 1.0 if in_wifi else 0.0,
    
    # --- ใส่ค่า Default สำหรับปัจจัยที่เราไม่ได้ให้ User กรอก ---
    # (ในการใช้งานจริง ควรใช้ค่าเฉลี่ย (mean) จาก Train Set)
    'latitude': 13.75, # ค่ากลางๆ กทม.
    'longitude': 100.5, # ค่ากลางๆ กทม.
    'proj_area': 2000.0, # สมมติค่าเฉลี่ย
})

# แปลง Dict เป็น DataFrame (1 แถว)
input_df = pd.DataFrame([input_data], columns=ALL_FEATURES)

# --- 5. ทำนายและแสดงผล ---
if st.button("ประเมินราคา (Predict)", use_container_width=True, type="primary"):
    
    # ส่ง DataFrame (1 แถว) เข้าไปทำนาย
    # โมเดล (Pipeline) จะทำ Pre-processing (Scale/Encode) ให้เอง
    prediction = model.predict(input_df)
    
    price = prediction[0]
    
    st.success(f"🎉 ราคาประเมิน (Value) คือ: {price:,.2f} บาท")
    
    # (ทางเลือก) แสดงข้อมูลที่ป้อนเข้าไป
    with st.expander("แสดงข้อมูลที่ใช้ในการทำนาย (Input DataFrame)"):
        st.dataframe(input_df)