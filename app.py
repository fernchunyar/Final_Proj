# Create a guideline page
def guideline_page():
    st.title("Guidelines for Using the Breast Cancer Classification App")

    # Create a more modern language selection dropdown with Thai as the default
    lang_button = st.selectbox("เลือกภาษา", ("Thai", "English"))  # Start with Thai

    if lang_button == "English":
        st.markdown(
            """
            This app classifies breast ultrasound images into two categories:
            **Benign** or **Malignant**. Follow these steps to use the app:
            """
        )
        # Display steps in English
        st.markdown(
            """
            ### Steps:
            1. Go to the **Classification** page using the navigation bar.
            2. Upload a breast ultrasound image using the **Upload Image** button.
            3. Wait for the app to process the image and display the results:
               - **Classification Result**: Shows whether the image is Benign or Malignant.
               - **Predicted Probabilities**: Displays the likelihood of each class.
            4. Repeat the process to classify another image.
            """
        )

        # Add an example image for visual aid
        st.image("intro_img.png", caption="Example Output", use_container_width=True)

        st.markdown(
            """
            ### Notes:
            - Ensure that the image is a valid breast ultrasound scan in `.jpg`, `.jpeg`, or `.png` format.
            - The app uses a pre-trained model and may take a few seconds to process your image.
            - For best results, upload clear and high-quality images.
            """
        )

        st.success("You are now ready to proceed to the **Classification** page!")
    
    elif lang_button == "Thai":
        st.markdown(
            """
            แอปนี้จำแนกภาพอัลตราซาวด์เต้านมเป็นสองประเภท:
            **Benign (ไม่เป็นมะเร็ง)** หรือ **Malignant (เป็นมะเร็ง)** โดยมีขั้นตอนการใช้งานแอปดังนี้:
            """
        )
        # Display steps in Thai
        st.markdown(
            """
            ### ขั้นตอน:
            1. ไปที่หน้า **Classification** ผ่านแถบเมนู
            2. อัปโหลดภาพอัลตราซาวด์เต้านมโดยใช้ปุ่ม **Upload Image**
            3. รอให้แอปประมวลผลภาพและแสดงผลลัพธ์:
               - **Classification Result**: แสดงว่าเป็น Benign หรือ Malignant
               - **Predicted Probabilities**: แสดงความน่าจะเป็นของแต่ละประเภท
            4. ทำซ้ำเพื่อจัดประเภทภาพอื่น
            """
        )

        # Add an example image for visual aid
        st.image("intro_img.png", caption="ตัวอย่างผลลัพธ์", use_container_width=True)

        st.markdown(
            """
            ### หมายเหตุ:
            - ตรวจสอบให้แน่ใจว่าภาพเป็นภาพอัลตราซาวด์เต้านมที่ถูกต้องในรูปแบบ `.jpg`, `.jpeg`, หรือ `.png`
            - แอปใช้โมเดลที่ผ่านการฝึกฝนแล้ว อาจใช้เวลาสักครู่ในการประมวลผลภาพของคุณ
            - สำหรับผลลัพธ์ที่ดีที่สุด ควรอัปโหลดภาพที่ชัดเจนและมีคุณภาพสูง
            """
        )

        st.success("ตอนนี้คุณพร้อมที่จะไปที่หน้า **Classification** แล้ว!")
