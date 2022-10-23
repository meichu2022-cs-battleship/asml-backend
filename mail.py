from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import codecs
import smtplib
from datetime import datetime


def sending_email(name, receiver, comment, kneck, bridge):

    content = MIMEMultipart()  # 建立MIMEMultipart物件
    content["subject"] = "Wafer Detect Report"  # 郵件標題
    content["from"] = "linus22765566@gmail.com"  # 寄件者
    content["to"] = receiver  # 收件者
    result = datetime.now().strftime("%Y-%m-%d %H:%M:%S %p")
    html = codecs.open("hero.html", "r").read()  # str
    html = html.replace("Name", name).replace("user_comment", comment)
    html = html.replace("kneck+bridge", str(kneck + bridge)).replace("TTT", result)
    html = html.replace("kneck", str(kneck)).replace("bridge", str(bridge))

    content.attach(MIMEText(html, "html"))  # 郵件內容
    password = "xrmnqflnsmlwtply"
    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:  # 設定SMTP伺服器
        try:

            smtp.ehlo()  # 驗證SMTP伺服器
            smtp.starttls()  # 建立加密傳輸
            smtp.login("linus22765566@gmail.com", password)  # 登入寄件者gmail

            fp = open("ASML.jpg", "rb")
            msgImage = MIMEImage(fp.read())
            fp.close()
            msgImage.add_header("Content-ID", "<ASML>")
            content.attach(msgImage)
            print("before origin.png")
            fp = open("origin.png", "rb")
            msgImage = MIMEImage(fp.read())
            fp.close()
            msgImage.add_header("Content-ID", "<origin>")
            content.attach(msgImage)

            fp = open("sem_gds_rg_rect.png", "rb")
            msgImage = MIMEImage(fp.read())
            fp.close()
            msgImage.add_header("Content-ID", "<after>")
            content.attach(msgImage)

            smtp.send_message(content)  # 寄送郵件
            print("Complete!")
        except Exception as e:
            print("Error message: ", e)
    return
